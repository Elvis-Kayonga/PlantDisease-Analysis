"""
Data preprocessing and augmentation module for plant disease detection.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Image preprocessing constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CHANNELS = 3


def load_images_from_directory(directory_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load images from directory with label inference from folder structure.
    Expected structure: directory_path/class_label/image_files
    
    Args:
        directory_path: Path to root directory containing class folders
        img_size: Target image size as (height, width)
    
    Returns:
        images: numpy array of images
        labels: numpy array of labels
        class_names: list of class names
    """
    images = []
    labels = []
    class_names = []
    label_idx = 0
    
    directory = Path(directory_path)
    if not directory.exists():
        logger.warning(f"Directory {directory_path} does not exist. Returning empty arrays.")
        return np.array([]), np.array([]), []
    
    # Iterate through class directories
    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_names.append(class_dir.name)
        logger.info(f"Loading class: {class_dir.name}")
        
        # Load images from class directory
        for img_file in class_dir.glob('*.jpg'):
            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_file, target_size=img_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                logger.warning(f"Failed to load image {img_file}: {e}")
        
        for img_file in class_dir.glob('*.JPG'):
            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_file, target_size=img_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                logger.warning(f"Failed to load image {img_file}: {e}")
        
        label_idx += 1
    
    return np.array(images), np.array(labels), class_names


def split_dataset(images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                  random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        train_ratio: fraction for training (default 0.7)
        val_ratio: fraction for validation (default 0.15)
        test_ratio: fraction for testing (default 0.15)
        random_state: random seed for reproducibility
    
    Returns:
        train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    # First split: train + temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val and test from temp
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=y_temp
    )
    
    logger.info(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_augmentation():
    """
    Create data augmentation pipeline for training.
    
    Returns:
        ImageDataGenerator instance
    """
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )


def create_data_generators(X_train, y_train, X_val, y_val, num_classes,
                          batch_size=BATCH_SIZE):
    """
    Create training and validation data generators with augmentation.
    
    Args:
        X_train, X_val: training and validation images
        y_train, y_val: training and validation labels
        num_classes: number of output classes
        batch_size: batch size for generators
    
    Returns:
        train_generator, val_generator
    """
    # One-hot encode labels
    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes)
    
    # Create augmentation for training
    train_augment = create_data_augmentation()
    
    # No augmentation for validation (just rescaling if needed)
    val_augment = ImageDataGenerator()
    
    train_generator = train_augment.flow(
        X_train, y_train_encoded,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_augment.flow(
        X_val, y_val_encoded,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator


def prepare_test_data(X_test, y_test, num_classes):
    """
    Prepare test data (one-hot encoded).
    
    Args:
        X_test: test images
        y_test: test labels
        num_classes: number of classes
    
    Returns:
        X_test, y_test_encoded
    """
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes)
    return X_test, y_test_encoded


def preprocess_single_image(image_path, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: path to image file
        img_size: target image size (height, width)
    
    Returns:
        preprocessed image array
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def balance_dataset(images, labels):
    """
    Check and report class balance in dataset.
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
    
    Returns:
        tuple (unique_labels, counts)
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  Class {label}: {count} samples")
    return unique_labels, counts
