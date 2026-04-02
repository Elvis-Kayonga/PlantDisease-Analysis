"""
Prediction module for making predictions with trained model.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

IMG_HEIGHT = 128
IMG_WIDTH = 128


class PlantDiseasePredictor:
    """
    Predictor class for plant disease classification.
    """
    
    def __init__(self, model_path, class_names):
        """
        Initialize predictor with trained model and class names.
        
        Args:
            model_path: path to saved model
            class_names: list of class names in order
        """
        self.model = None
        self.saved_model = None
        self.saved_model_signature = None
        self.model_kind = None
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else 0
        logger.info(f"Classes: {self.class_names}")

    def _candidate_model_paths(self):
        """Return model paths to try in order of preference."""
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            self.model_path,
            self.model_path.with_suffix(".keras"),
            self.model_path.with_name(self.model_path.stem),
            repo_root / "notebook" / "models" / "plant_disease_model",
            repo_root / "notebook" / "models" / "plant_disease_model.keras",
            repo_root / "notebook" / "models" / "plant_disease_model.h5",
        ]

        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in seen:
                seen.add(candidate_str)
                ordered_candidates.append(candidate)

        return ordered_candidates

    def _load_saved_model(self, model_dir):
        """Load a TensorFlow SavedModel directory for inference."""
        self.saved_model = tf.saved_model.load(str(model_dir))
        signatures = getattr(self.saved_model, "signatures", {}) or {}
        self.saved_model_signature = signatures.get("serving_default")
        if self.saved_model_signature is None and signatures:
            self.saved_model_signature = next(iter(signatures.values()))
        self.model_kind = "saved_model"

    def _load_keras_model(self, model_file):
        """Load a Keras model file without compiling it."""
        self.model = tf.keras.models.load_model(str(model_file), compile=False)
        self.model_kind = "keras"

    def _try_override_classes(self, candidate_path):
        """If returning a fallback model, prefer the 'class_names.pkl' located beside it."""
        class_pkl = candidate_path.parent / "class_names.pkl"
        if class_pkl.exists():
            import pickle
            try:
                with open(class_pkl, 'rb') as f:
                    new_classes = pickle.load(f)
                if new_classes and len(new_classes) != self.num_classes:
                    logger.info(f"Overriding class_names from {class_pkl} (found {len(new_classes)} classes, previous was {self.num_classes})")
                    self.class_names = new_classes
                    self.num_classes = len(new_classes)
            except Exception as e:
                logger.warning(f"Failed to load class overlay from {class_pkl}: {e}")

    def _load_model_artifact(self):
        """Load the first compatible model artifact we can find."""
        load_errors = []

        for candidate in self._candidate_model_paths():
            try:
                if candidate.exists() and candidate.is_dir():
                    self._load_saved_model(candidate)
                    logger.info(f"Loaded SavedModel from {candidate}")
                    self._try_override_classes(candidate)
                    return

                if candidate.exists() and candidate.is_file():
                    self._load_keras_model(candidate)
                    logger.info(f"Loaded Keras model from {candidate}")
                    self._try_override_classes(candidate)
                    return
            except Exception as exc:
                load_errors.append(f"{candidate}: {exc}")

        raise RuntimeError(
            "Unable to load any model artifact. Tried: " + " | ".join(load_errors)
        )

    def _predict_with_saved_model(self, img_batch):
        """Run inference through a SavedModel signature."""
        if self.saved_model_signature is None:
            raise RuntimeError("SavedModel signature is not available")

        input_signature = self.saved_model_signature.structured_input_signature[1]
        tensor_input = tf.convert_to_tensor(img_batch)

        if input_signature:
            input_name = next(iter(input_signature.keys()))
            outputs = self.saved_model_signature(**{input_name: tensor_input})
        else:
            outputs = self.saved_model_signature(tensor_input)

        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        elif isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        return outputs.numpy()

    def _predict_batch(self, img_batch):
        """Predict probabilities from whichever model backend was loaded."""
        if self.model_kind == "saved_model":
            return self._predict_with_saved_model(img_batch)

        if self.model is None:
            raise RuntimeError("Model backend is not available")

        return self.model.predict(img_batch, verbose=0)
    
    def predict_image(self, image_path, return_all_probs=False):
        """
        Predict class for a single image.
        
        Args:
            image_path: path to image file
            return_all_probs: if True, return probabilities for all classes
        
        Returns:
            dict with predicted_class, confidence, and optionally all_probabilities
        """
        try:
            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self._predict_batch(img_array)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            result = {
                'predicted_class': self.class_names[predicted_idx],
                'predicted_idx': int(predicted_idx),
                'confidence': confidence
            }
            
            if return_all_probs:
                result['all_probabilities'] = {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(self.num_classes)
                }
            
            logger.info(f"Prediction for {image_path}: {result['predicted_class']} "
                       f"(confidence: {confidence:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            raise
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: list of image file paths
        
        Returns:
            list of prediction dicts
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, return_all_probs=True)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction for {image_path}: {e}")
                results.append({'error': str(e), 'image': image_path})
        
        return results
    
    def predict_array(self, image_array):
        """
        Predict class for a numpy array image.
        
        Args:
            image_array: numpy array of shape (H, W, 3) with values in [0, 255] or [0, 1]
        
        Returns:
            prediction dict
        """
        try:
            # Resize to model input size
            if image_array.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
                img_resized = tf.image.resize(image_array, (IMG_HEIGHT, IMG_WIDTH))
            else:
                img_resized = image_array
            
            # Normalize if needed
            if np.max(img_resized) > 1.0:
                img_resized = img_resized / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_resized, axis=0)
            
            # Predict
            predictions = self._predict_batch(img_batch)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx])
            
            result = {
                'predicted_class': self.class_names[predicted_idx],
                'predicted_idx': int(predicted_idx),
                'confidence': confidence,
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(self.num_classes)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting array: {e}")
            raise
    
    def evaluate_on_dataset(self, X_test, y_test):
        """
        Evaluate model on test dataset.
        
        Args:
            X_test: test images
            y_test: test labels (one-hot encoded)
        
        Returns:
            dict with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Evaluation requires a compiled Keras model file")

        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        evaluation = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]) if len(results) > 2 else None,
            'recall': float(results[3]) if len(results) > 3 else None
        }
        
        logger.info(f"Evaluation metrics: {evaluation}")
        return evaluation


def get_predictions_for_directory(model_path, class_names, image_dir):
    """
    Get predictions for all images in a directory.
    
    Args:
        model_path: path to saved model
        class_names: list of class names
        image_dir: directory containing images
    
    Returns:
        list of prediction dicts
    """
    predictor = PlantDiseasePredictor(model_path, class_names)
    image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.JPG'))
    return predictor.predict_batch(image_paths)
