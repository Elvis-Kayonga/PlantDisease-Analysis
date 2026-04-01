"""
Configuration file for Plant Disease Detection System.
Centralizes all configuration parameters.
"""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
UPLOADED_DATA_DIR = BASE_DIR / "uploaded_data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, UPLOADED_DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 5  # Change to 5-8 based on your dataset

# Training Configuration
MAX_EPOCHS = 8
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
NUM_LAYERS_TO_UNFREEZE = 10
MAX_IMAGES_PER_CLASS = 1200  # ~6000 total for 5 classes

# Data Processing
TRAIN_TEST_SPLIT = 0.70
VAL_TEST_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Dataset Configuration
DATA_AUGMENTATION = {
    'rotation_range': 25,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model Paths
MODEL_PATH = str(MODELS_DIR / "plant_disease_model.h5")
MODEL_SAVEDMODEL_PATH = str(MODELS_DIR / "plant_disease_model")
METADATA_PATH = str(MODELS_DIR / "metadata.json")
CLASS_NAMES_PATH = str(MODELS_DIR / "class_names.pkl")

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
API_RELOAD = False
API_LOG_LEVEL = "info"

# Streamlit Configuration
STREAMLIT_HOST = "0.0.0.0"
STREAMLIT_PORT = 8501
STREAMLIT_THEME = "light"
STREAMLIT_LOGGER = "info"

# Load Testing Configuration
LOCUST_WAIT_TIME_MIN = 1
LOCUST_WAIT_TIME_MAX = 3
LOCUST_SPAWN_RATE = 1
DEFAULT_NUM_USERS = 10
DEFAULT_TEST_DURATION = 60

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = str(LOGS_DIR / "app.log")

# Kaggle Configuration (for dataset download)
KAGGLE_DATASET = "vipoooool/new-plant-diseases-dataset"
KAGGLE_DATASET_PATH = "/tmp/plant_disease_dataset"

# GPU Configuration
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8

# Production Configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

# Retraining Configuration
AUTO_RETRAIN_THRESHOLD = 100  # Retrain when this many images uploaded
KEEP_OLD_MODELS = True
MAX_SAVED_MODELS = 5

# Security
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ALGORITHM = "HS256"
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

# Database (if using in future)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Print configuration summary
if __name__ == "__main__":
    print("Plant Disease Detection System - Configuration")
    print("=" * 50)
    print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print(f"Training Split: {TRAIN_TEST_SPLIT}")
    print(f"Validation Split: {VAL_TEST_SPLIT}")
    print(f"Test Split: {TEST_SPLIT}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"API Port: {API_PORT}")
    print(f"Streamlit Port: {STREAMLIT_PORT}")
    print("=" * 50)
