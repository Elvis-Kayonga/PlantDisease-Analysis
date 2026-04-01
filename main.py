"""
FastAPI backend for plant disease detection model.
Provides endpoints for prediction, data upload, and retraining.
"""

import sys
sys.path.insert(0, r"C:\tfpkg")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime
from time import perf_counter
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import re
import io

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image, UnidentifiedImageError

from src.preprocessing import split_dataset
from src.model import create_model, compile_model, train_model_on_arrays, fine_tune_model, save_model
from src.prediction import PlantDiseasePredictor
from src.database import (
    init_db,
    log_prediction,
    log_upload,
    log_training_run,
    get_database_status,
    get_prediction_history,
    get_training_history,
    upsert_classes,
    safe_call,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for plant disease classification and model retraining",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
MODEL_PATH = "models/plant_disease_model.h5"
METADATA_PATH = "models/metadata.json"
CLASS_NAMES_PATH = "models/class_names.pkl"
UPLOADED_DATA_DIR = "uploaded_data"
MODELS_DIR = "models"
DB_PATH = "data/plant_disease.db"

# Global variables for model and metadata
predictor = None
class_names = None
metadata = None

# Model state
model_status = {
    "loaded": False,
    "last_trained": None,
    "training_in_progress": False,
    "accuracy": None,
    "total_samples": 0
}


def load_model_resources():
    """Load model and metadata from disk."""
    global predictor, class_names, metadata
    
    try:
        # Load class names
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'rb') as f:
                class_names = pickle.load(f)
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        
        # Load model
        if os.path.exists(MODEL_PATH):
            predictor = PlantDiseasePredictor(MODEL_PATH, class_names)
            model_status["loaded"] = True
            model_status["accuracy"] = (metadata or {}).get("metrics", {}).get("accuracy")
            model_status["last_trained"] = (metadata or {}).get("last_trained")
            model_status["total_samples"] = (metadata or {}).get("total_samples", 0)
            safe_call(upsert_classes, class_names or [], source="model", db_path=DB_PATH)
            logger.info("Model and metadata loaded successfully")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading model resources: {e}")


def _safe_filename(file_name: str) -> str:
    """Normalize a potentially unsafe filename to a filesystem-safe name."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", file_name)
    return normalized or f"upload_{int(datetime.now().timestamp())}.jpg"


def _normalize_class_name(class_name: str) -> str:
    """Normalize class name to a stable folder-safe variant."""
    cleaned = class_name.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", cleaned)
    if not cleaned:
        raise ValueError("Class name cannot be empty after normalization")
    return cleaned


def _class_slug(value: str) -> str:
    """Create a compare-friendly class slug."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _canonicalize_class_name(class_name: str) -> str:
    """Map free-form class labels to canonical model class names when possible."""
    normalized = _normalize_class_name(class_name)

    alias_map = {
        "cedarapplerust": "Apple___Cedar_apple_rust",
        "applecedarrust": "Apple___Cedar_apple_rust",
        "applecedarapplerust": "Apple___Cedar_apple_rust",
        "applescab": "Apple___Apple_scab",
        "appleapplescab": "Apple___Apple_scab",
    }

    candidate_slug = _class_slug(normalized)
    mapped_target = alias_map.get(candidate_slug, normalized)

    # If current model classes are known, try exact and token-based canonical match.
    available_classes = class_names or []
    if available_classes:
        by_slug = {_class_slug(name): name for name in available_classes}
        for target in [mapped_target, normalized]:
            target_slug = _class_slug(target)
            if target_slug in by_slug:
                return by_slug[target_slug]

        tokens = set(re.findall(r"[a-z0-9]+", mapped_target.lower()))
        if tokens:
            token_matches = []
            for known in available_classes:
                known_tokens = set(re.findall(r"[a-z0-9]+", known.lower()))
                if tokens.issubset(known_tokens):
                    token_matches.append(known)
            if len(token_matches) == 1:
                return token_matches[0]

    return mapped_target


def _extract_class_from_filename(file_name: str) -> Optional[Tuple[str, str]]:
    """
    Extract class from filename convention: class__image.jpg.
    Returns tuple(class_name, clean_filename) or None.
    """
    if "__" not in file_name:
        return None

    raw_class, raw_name = file_name.split("__", 1)
    if not raw_class.strip() or not raw_name.strip():
        return None

    return _canonicalize_class_name(raw_class), _safe_filename(raw_name)


def _load_images_with_class_map(
    directory_path: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    include_new_classes: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load images from class subfolders while preserving global class mapping."""
    images: List[np.ndarray] = []
    labels: List[int] = []

    if class_to_idx is None:
        class_to_idx = {}

    if not os.path.exists(directory_path):
        return np.array(images), np.array(labels), class_to_idx

    for class_name in sorted(os.listdir(directory_path)):
        class_dir = os.path.join(directory_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        if class_name not in class_to_idx:
            if not include_new_classes:
                continue
            class_to_idx[class_name] = len(class_to_idx)

        class_idx = class_to_idx[class_name]

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(class_dir, file_name)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    image_path,
                    target_size=(IMG_HEIGHT, IMG_WIDTH)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Skipping image {image_path}: {e}")

    return np.array(images), np.array(labels), class_to_idx


def _combine_datasets(image_arrays: List[np.ndarray], label_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Safely combine potentially empty arrays from multiple sources."""
    valid_pairs = [
        (images, labels)
        for images, labels in zip(image_arrays, label_arrays)
        if len(images) > 0 and len(labels) > 0
    ]

    if not valid_pairs:
        return np.array([]), np.array([])

    images = np.concatenate([pair[0] for pair in valid_pairs], axis=0)
    labels = np.concatenate([pair[1] for pair in valid_pairs], axis=0)
    return images, labels


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up Plant Disease Detection API")
    safe_call(init_db, DB_PATH)
    load_model_resources()
    logger.info("API startup completed")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_status["loaded"],
        "api_version": "1.0.0"
    }


@app.get("/model-status")
async def get_model_status():
    """Get current model status."""
    return {
        "loaded": model_status["loaded"],
        "accuracy": model_status["accuracy"],
        "last_trained": model_status["last_trained"],
        "training_in_progress": model_status["training_in_progress"],
        "total_samples_trained_on": model_status["total_samples"],
        "classes": class_names,
        "num_classes": len(class_names) if class_names else 0
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        Prediction result with class label and confidence
    """
    if not model_status["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        started = perf_counter()
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Decode safely from bytes; supports common web image formats.
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Unsupported or corrupted image file. Please upload a valid image (jpg, jpeg, png, webp)."
            )

        pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(pil_img, dtype=np.float32) / 255.0

        # Make prediction directly from array to avoid temp-file handling issues.
        result = predictor.predict_array(img_array)
        latency_ms = (perf_counter() - started) * 1000

        safe_call(
            log_prediction,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["all_probabilities"],
            file_name=file.filename,
            latency_ms=latency_ms,
            source="api",
            db_path=DB_PATH,
        )
        low_conf_threshold = 0.55
        
        return {
            "success": True,
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "all_probabilities": result['all_probabilities'],
            "low_confidence_warning": result['confidence'] < low_conf_threshold,
            "latency_ms": round(latency_ms, 2)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/upload-data")
async def upload_data(
    files: List[UploadFile] = File(...),
    class_name: Optional[str] = Form(None)
):
    """
    Upload images for retraining.
    
    Args:
        files: List of image files
    
    Returns:
        Upload status
    """
    try:
        os.makedirs(UPLOADED_DATA_DIR, exist_ok=True)
        uploaded_count = 0
        skipped_files: List[str] = []
        class_counts: Dict[str, int] = {}

        normalized_class_name = _canonicalize_class_name(class_name) if class_name else None
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                logger.warning(f"Skipping non-image file: {file.filename}")
                skipped_files.append(file.filename)
                continue

            file_class_name = normalized_class_name
            safe_name = _safe_filename(file.filename)

            if not file_class_name:
                extracted = _extract_class_from_filename(file.filename)
                if extracted:
                    file_class_name, safe_name = extracted

            if not file_class_name:
                skipped_files.append(file.filename)
                logger.warning(
                    "Skipping file without class label. Use class_name form field "
                    "or filename format class__filename.jpg"
                )
                continue

            class_dir = os.path.join(UPLOADED_DATA_DIR, file_class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Save file
            file_path = os.path.join(class_dir, safe_name)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            uploaded_count += 1
            class_counts[file_class_name] = class_counts.get(file_class_name, 0) + 1
            logger.info(f"Uploaded: {file.filename} -> {file_class_name}/{safe_name}")

        total_uploaded_data = 0
        for root, _, file_names in os.walk(UPLOADED_DATA_DIR):
            total_uploaded_data += len([f for f in file_names if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        safe_call(
            log_upload,
            class_name=normalized_class_name,
            uploaded_count=uploaded_count,
            skipped_files=skipped_files,
            total_uploaded_data=total_uploaded_data,
            db_path=DB_PATH,
        )
        
        return {
            "success": True,
            "uploaded_count": uploaded_count,
            "skipped_files": skipped_files,
            "class_distribution": class_counts,
            "total_uploaded_data": total_uploaded_data
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """
    Trigger model retraining with accumulated data.
    
    Returns:
        Retraining status
    """
    if model_status["training_in_progress"]:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    try:
        model_status["training_in_progress"] = True
        logger.info("Starting model retraining...")
        safe_call(
            log_training_run,
            status="started",
            message="Retraining started",
            db_path=DB_PATH,
        )

        # Validate uploaded data exists and has class-labeled structure.
        if not os.path.exists(UPLOADED_DATA_DIR):
            raise HTTPException(status_code=400, detail="No uploaded_data directory found")

        uploaded_class_dirs = [
            d for d in os.listdir(UPLOADED_DATA_DIR)
            if os.path.isdir(os.path.join(UPLOADED_DATA_DIR, d))
        ]
        if not uploaded_class_dirs:
            raise HTTPException(
                status_code=400,
                detail="No class-labeled uploaded data found. Upload with class_name or class__filename format."
            )

        # Build class mapping from existing model metadata if available.
        class_to_idx: Dict[str, int] = {}
        if class_names:
            for idx, class_name in enumerate(class_names):
                class_to_idx[class_name] = idx

        # Load baseline training data and newly uploaded data.
        X_base, y_base, class_to_idx = _load_images_with_class_map(
            "data/train",
            class_to_idx=class_to_idx,
            include_new_classes=False
        )
        X_uploaded, y_uploaded, class_to_idx = _load_images_with_class_map(
            UPLOADED_DATA_DIR,
            class_to_idx=class_to_idx,
            include_new_classes=True
        )

        X_all, y_all = _combine_datasets([X_base, X_uploaded], [y_base, y_uploaded])
        if len(X_all) < 20:
            raise HTTPException(
                status_code=400,
                detail="Not enough images to retrain. Provide at least 20 labeled images."
            )

        unique_classes = np.unique(y_all)
        if len(unique_classes) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 classes for retraining."
            )

        # Deterministic class order for metadata/prediction mapping.
        sorted_class_items = sorted(class_to_idx.items(), key=lambda item: item[1])
        resolved_class_names = [item[0] for item in sorted_class_items]
        num_classes = len(resolved_class_names)

        # Split dataset (with stratification fallback for small classes).
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                X_all,
                y_all,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_state=42
            )
        except Exception as split_error:
            logger.warning(f"Stratified split failed, using random split fallback: {split_error}")
            indices = np.arange(len(X_all))
            np.random.shuffle(indices)
            X_all = X_all[indices]
            y_all = y_all[indices]

            train_end = int(0.7 * len(X_all))
            val_end = int(0.85 * len(X_all))

            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
            X_test, y_test = X_all[val_end:], y_all[val_end:]

        y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes)

        # Train transfer-learning model.
        model, base_model = create_model(num_classes=num_classes, freeze_base=True)
        compile_model(model, learning_rate=0.001)

        train_model_on_arrays(
            model,
            X_train,
            y_train_encoded,
            X_val,
            y_val_encoded,
            epochs=5,
            batch_size=BATCH_SIZE,
            model_checkpoint_path=MODEL_PATH
        )

        # Light fine-tune pass.
        fine_tune_model(
            model,
            base_model,
            num_layers_to_unfreeze=10,
            learning_rate=0.0001
        )
        train_model_on_arrays(
            model,
            X_train,
            y_train_encoded,
            X_val,
            y_val_encoded,
            epochs=2,
            batch_size=BATCH_SIZE,
            model_checkpoint_path=MODEL_PATH
        )

        # Evaluate on holdout split.
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        }

        save_model(model, MODEL_PATH)
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(CLASS_NAMES_PATH, 'wb') as class_file:
            pickle.dump(resolved_class_names, class_file)

        new_metadata = {
            "model_type": "MobileNetV2_Transfer_Learning",
            "last_trained": datetime.now().isoformat(),
            "total_samples": int(len(X_all)),
            "classes": resolved_class_names,
            "metrics": metrics,
            "training_config": {
                "image_size": [IMG_HEIGHT, IMG_WIDTH],
                "batch_size": BATCH_SIZE,
                "epochs": 7,
                "fine_tune_layers": 10
            }
        }
        with open(METADATA_PATH, 'w', encoding='utf-8') as metadata_file:
            json.dump(new_metadata, metadata_file, indent=2)

        # Refresh in-memory resources.
        load_model_resources()
        safe_call(upsert_classes, resolved_class_names, source="retrain", db_path=DB_PATH)
        safe_call(
            log_training_run,
            status="completed",
            message="Model retraining completed",
            total_samples=len(X_all),
            num_classes=num_classes,
            metrics=metrics,
            details={"uploaded_classes": uploaded_class_dirs},
            db_path=DB_PATH,
        )
        model_status["training_in_progress"] = False

        return {
            "success": True,
            "message": "Model retraining completed",
            "last_trained": new_metadata["last_trained"],
            "metrics": metrics,
            "num_classes": num_classes,
            "total_samples": len(X_all)
        }
        
    except Exception as e:
        model_status["training_in_progress"] = False
        logger.error(f"Retraining error: {e}")
        safe_call(
            log_training_run,
            status="failed",
            message=str(e),
            db_path=DB_PATH,
        )
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return {
        "metrics": metadata.get("metrics", {}),
        "model_type": metadata.get("model_type"),
        "training_config": metadata.get("training_config"),
        "classes": metadata.get("classes")
    }


@app.get("/db-status")
async def db_status():
    """Get SQLite database usage statistics."""
    status = safe_call(get_database_status, db_path=DB_PATH) or {}
    return status


@app.get("/prediction-history")
async def prediction_history(limit: int = 50):
    """Get recent prediction logs."""
    items = safe_call(get_prediction_history, limit=limit, db_path=DB_PATH) or []
    return {"count": len(items), "items": items}


@app.get("/training-history")
async def training_history(limit: int = 30):
    """Get recent retraining runs."""
    items = safe_call(get_training_history, limit=limit, db_path=DB_PATH) or []
    return {"count": len(items), "items": items}


# --- Serve React Frontend ---
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")

if os.path.exists(FRONTEND_DIST):
    # Mount the assets directory specifically
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")
    
    # Serve the main index.html for the root path and any unhandled paths (SPA fallback)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Ignore API routes explicitly just in case
        if full_path.startswith("api/") or full_path in ["docs", "openapi.json"]:
            raise HTTPException(status_code=404, detail="Not Found")
            
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        requested_file = os.path.join(FRONTEND_DIST, full_path)
        
        if os.path.exists(requested_file) and os.path.isfile(requested_file):
            return FileResponse(requested_file)
            
        return FileResponse(index_file)
else:
    logger.warning("Frontend dist directory not found. Please build the React app (npm run build) to serve it from the API.")
# ----------------------------


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
