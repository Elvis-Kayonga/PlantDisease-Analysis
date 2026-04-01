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
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.num_classes = len(class_names)
        logger.info(f"Predictor initialized with model from {model_path}")
        logger.info(f"Classes: {self.class_names}")
    
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
            predictions = self.model.predict(img_array, verbose=0)
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
            predictions = self.model.predict(img_batch, verbose=0)
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
