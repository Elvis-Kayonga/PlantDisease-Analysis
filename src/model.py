"""
Model creation and training module using MobileNetV2 transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import os

logger = logging.getLogger(__name__)

IMG_HEIGHT = 224
IMG_WIDTH = 224


def create_model(num_classes, freeze_base=True):
    """
    Create a MobileNetV2-based model with custom top layers.
    
    Args:
        num_classes: number of output classes
        freeze_base: whether to freeze base model weights
    
    Returns:
        keras model
    """
    # Load pretrained MobileNetV2 (ImageNet weights)
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model if requested
    if freeze_base:
        base_model.trainable = False
        logger.info("Base model layers frozen for transfer learning")
    
    # Create new model with custom top layers
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer and loss function.
    
    Args:
        model: keras model to compile
        learning_rate: learning rate for optimizer
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )
    logger.info(f"Model compiled with Adam optimizer (lr={learning_rate})")


def get_callbacks(model_checkpoint_path=None):
    """
    Create training callbacks for early stopping and learning rate reduction.
    
    Args:
        model_checkpoint_path: path to save best model
    
    Returns:
        list of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    if model_checkpoint_path:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )
    
    return callbacks


def train_model(model, train_generator, val_generator, epochs=5, 
                model_checkpoint_path=None):
    """
    Train the model with data generators.
    
    Args:
        model: keras model to train
        train_generator: training data generator
        val_generator: validation data generator
        epochs: number of epochs to train
        model_checkpoint_path: path to save best model
    
    Returns:
        training history
    """
    callbacks = get_callbacks(model_checkpoint_path)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"Training completed for {epochs} epochs")
    return history


def train_model_on_arrays(model, X_train, y_train, X_val, y_val, epochs=5,
                          batch_size=32, model_checkpoint_path=None):
    """
    Train model on numpy arrays (alternative to generators).
    
    Args:
        model: keras model to train
        X_train, y_train: training data
        X_val, y_val: validation data
        epochs: number of epochs
        batch_size: batch size
        model_checkpoint_path: path to save best model
    
    Returns:
        training history
    """
    callbacks = get_callbacks(model_checkpoint_path)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def fine_tune_model(model, base_model, num_layers_to_unfreeze=10, 
                    learning_rate=0.0001):
    """
    Fine-tune the model by unfreezing some base model layers.
    
    Args:
        model: full model
        base_model: base MobileNetV2 model
        num_layers_to_unfreeze: number of layers to unfreeze from the end
        learning_rate: learning rate for fine-tuning
    """
    # Unfreeze last N layers of base model
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    
    logger.info(f"Unfroze last {num_layers_to_unfreeze} layers for fine-tuning")
    
    # Recompile with lower learning rate
    compile_model(model, learning_rate=learning_rate)
    logger.info(f"Model recompiled for fine-tuning with lr={learning_rate}")


def save_model(model, model_path):
    """
    Save model to disk.
    
    Args:
        model: keras model to save
        model_path: path to save model (should be .tf or .h5)
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load model from disk.
    
    Args:
        model_path: path to saved model
    
    Returns:
        loaded keras model
    """
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def get_model_summary(model):
    """
    Print and return model summary.
    
    Args:
        model: keras model
    """
    model.summary()
