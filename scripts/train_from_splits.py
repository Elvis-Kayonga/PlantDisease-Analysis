from pathlib import Path
import json
import pickle
from datetime import datetime
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model, compile_model, train_model, fine_tune_model, save_model

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
SEED = 42
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 2

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/validation")
TEST_DIR = Path("data/test")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "plant_disease_model.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


def make_dataset(path: Path, shuffle: bool):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )
    class_names = list(ds.class_names)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def to_one_hot_dataset(ds, num_classes: int):
    return ds.map(
        lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def collect_labels_and_predictions(model, ds):
    y_true = []
    y_pred = []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def count_images_in_split(path: Path) -> int:
    return len([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def main():
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        raise RuntimeError("Training directory is empty. Run dataset preparation first.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_train_ds, class_names = make_dataset(TRAIN_DIR, shuffle=True)
    raw_val_ds, _ = make_dataset(VAL_DIR, shuffle=False)
    raw_test_ds, _ = make_dataset(TEST_DIR, shuffle=False)

    num_classes = len(class_names)

    train_ds = to_one_hot_dataset(raw_train_ds, num_classes)
    val_ds = to_one_hot_dataset(raw_val_ds, num_classes)

    model, base_model = create_model(num_classes=num_classes, freeze_base=True)
    compile_model(model, learning_rate=0.001)

    # Baseline transfer-learning fit.
    train_model(model, train_ds, val_ds, epochs=PHASE1_EPOCHS, model_checkpoint_path=str(MODEL_PATH))

    # Light fine-tuning pass.
    fine_tune_model(model, base_model, num_layers_to_unfreeze=10, learning_rate=0.0001)
    train_model(model, train_ds, val_ds, epochs=PHASE2_EPOCHS, model_checkpoint_path=str(MODEL_PATH))

    # Evaluate on integer-label test set.
    y_true, y_pred = collect_labels_and_predictions(model, raw_test_ds)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    save_model(model, str(MODEL_PATH))

    with open(CLASS_NAMES_PATH, "wb") as class_file:
        pickle.dump(class_names, class_file)

    total_samples = (
        count_images_in_split(TRAIN_DIR)
        + count_images_in_split(VAL_DIR)
        + count_images_in_split(TEST_DIR)
    )

    metadata = {
        "model_type": "MobileNetV2_Transfer_Learning",
        "last_trained": datetime.now().isoformat(),
        "total_samples": int(total_samples),
        "classes": class_names,
        "metrics": metrics,
        "training_config": {
            "image_size": [IMG_HEIGHT, IMG_WIDTH],
            "batch_size": BATCH_SIZE,
            "epochs": PHASE1_EPOCHS + PHASE2_EPOCHS,
            "fine_tune_layers": 10,
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print("Training complete.")
    print("Classes:", class_names)
    print("Metrics:", metrics)
    print("Saved:", MODEL_PATH, CLASS_NAMES_PATH, METADATA_PATH)


if __name__ == "__main__":
    main()
