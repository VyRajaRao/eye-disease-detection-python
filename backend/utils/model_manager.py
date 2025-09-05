"""
Model Manager for Eye Disease Detection
Handles CNN model creation, training, loading, and prediction using TensorFlow/Keras
"""

import os
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages CNN model for eye disease classification
    Handles model creation, training, loading, saving, and prediction
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize the model manager
        
        Args:
            config: Configuration object with model settings
        """
        self.config = config
        self.model = None
        self.training_history = None
        self.class_names = config.CLASS_NAMES
        self.model_info = {
            'name': 'EyeDiseaseClassifier',
            'version': '1.0.0',
            'input_shape': (*config.TARGET_IMAGE_SIZE, 3),
            'num_classes': len(config.CLASS_NAMES),
            'trained': False,
            'training_date': None
        }
    
    def create_model(self, input_shape: Tuple[int, int, int] = None) -> keras.Model:
        """
        Create CNN model architecture for eye disease classification
        
        Args:
            input_shape: Input shape for the model (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
        if input_shape is None:
            input_shape = (*self.config.TARGET_IMAGE_SIZE, 3)
        
        try:
            # Create Sequential model
            model = models.Sequential([
                # Input layer
                layers.Input(shape=input_shape),
                
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fourth Convolutional Block
                layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='last_conv'),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Global Average Pooling (instead of Flatten to reduce parameters)
                layers.GlobalAveragePooling2D(),
                
                # Dense layers
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                # Output layer
                layers.Dense(len(self.class_names), activation='softmax', name='predictions')
            ])
            
            # Compile the model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info(f"Model created with input shape: {input_shape}")
            logger.info(f"Total parameters: {model.count_params()}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load trained model from disk
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = self.config.MODEL_PATH
            
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
                
                # Update model info
                self.model_info['trained'] = True
                
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                # Create a basic model if no trained model exists
                self.model = self.create_model()
                # Build the model by making a dummy prediction
                dummy_input = np.zeros((1, *self.config.TARGET_IMAGE_SIZE, 3), dtype=np.float32)
                _ = self.model(dummy_input)
                logger.info("Model built with dummy input")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Create a basic model as fallback
            self.model = self.create_model()
            # Build the model by making a dummy prediction
            dummy_input = np.zeros((1, *self.config.TARGET_IMAGE_SIZE, 3), dtype=np.float32)
            _ = self.model(dummy_input)
            logger.info("Fallback model built with dummy input")
            return False
    
    def save_model(self, model_path: str = None) -> bool:
        """
        Save trained model to disk
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if model saved successfully, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            if model_path is None:
                model_path = self.config.MODEL_PATH
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save model info
            info_path = model_path.replace('.h5', '_info.pkl')
            with open(info_path, 'wb') as f:
                pickle.dump(self.model_info, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model is None:
                raise ValueError("No model loaded")
            
            start_time = time.time()
            
            # Make prediction
            predictions = self.model.predict(image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            processing_time = time.time() - start_time
            
            # Get all class predictions
            all_predictions = [
                {
                    'class': self.class_names[i],
                    'confidence': float(predictions[0][i])
                }
                for i in range(len(self.class_names))
            ]
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'processing_time': processing_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        return self.model_info.copy()
