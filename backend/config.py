"""
Configuration settings for Eye Disease Detection Backend
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class with performance optimizations"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Server settings with performance optimizations
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    THREADED = True  # Enable threading for better concurrency
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    # Performance settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes cache timeout
    JSON_SORT_KEYS = False  # Disable JSON key sorting for better performance
    
    # Request handling optimizations
    MAX_FORM_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB for form data
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'eye_disease_model.h5')
    MODEL_BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'backup')
    
    # Image processing settings
    TARGET_IMAGE_SIZE = (224, 224)  # Standard input size for most CNN models
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    
    # Class names for eye diseases
    CLASS_NAMES = [
        'Normal',
        'Diabetic Retinopathy',
        'Glaucoma',
        'Cataract',
        'Age-related Macular Degeneration',
        'Hypertensive Retinopathy'
    ]
    
    # Data augmentation settings
    AUGMENTATION_CONFIG = {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'zoom_range': 0.2,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    }
    
    # Visualization settings
    GRADCAM_LAYER_NAME = 'conv2d_4'  # Last convolutional layer name
    HEATMAP_ALPHA = 0.6
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings here

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
