"""
Eye Disease Detection Backend API
Flask application for AI-based eye disease prediction using local ML models
"""

import os
import io
import base64
import time
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import logging
from datetime import datetime

# Import custom modules
from utils.image_preprocessor import ImagePreprocessor
from utils.model_manager import ModelManager
from utils.visualization import VisualizationGenerator
from config import Config

# Initialize Flask app with performance optimizations
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins=['http://localhost:5173', 'http://localhost:8080', 'http://localhost:3000'])

# Initialize caching for better performance
cache = Cache(app)

# Performance monitoring decorator
def monitor_performance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Add performance metrics to response if it's a JSON response
        if hasattr(result, 'get_json') or isinstance(result, tuple):
            try:
                if isinstance(result, tuple):
                    response_data, status_code = result
                    if isinstance(response_data, dict) or hasattr(response_data, 'get_json'):
                        if hasattr(response_data, 'get_json'):
                            data = response_data.get_json()
                        else:
                            data = response_data
                        data['api_response_time'] = round((end_time - start_time) * 1000, 2)  # ms
                        logger.info(f"API {f.__name__} took {data['api_response_time']}ms")
                        return jsonify(data), status_code
                else:
                    data = result.get_json()
                    data['api_response_time'] = round((end_time - start_time) * 1000, 2)  # ms
                    logger.info(f"API {f.__name__} took {data['api_response_time']}ms")
                    return jsonify(data)
            except:
                logger.info(f"API {f.__name__} took {round((end_time - start_time) * 1000, 2)}ms")
                
        return result
    return decorated_function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
image_processor = ImagePreprocessor()
model_manager = ModelManager()
viz_generator = VisualizationGenerator()

# Load the trained model on startup
try:
    model_manager.load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")

@app.route('/api/health', methods=['GET'])
@monitor_performance
@cache.cached(timeout=60)  # Cache for 1 minute
def health_check():
    """Health check endpoint with performance monitoring"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_manager.is_model_loaded(),
        'cache_enabled': True,
        'threading_enabled': app.config.get('THREADED', False)
    }

@app.route('/api/predict', methods=['POST'])
@monitor_performance
def predict_eye_disease():
    """
    Predict eye disease from uploaded retinal image
    """
    try:
        # Check if model is loaded
        if not model_manager.is_model_loaded():
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate file type
        if not _allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please use JPG, PNG, or JPEG'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image for model input
        processed_image = image_processor.preprocess_for_prediction(np.array(image))
        
        # Make prediction
        prediction_results = model_manager.predict(processed_image)
        
        # Generate visualization (heatmap) if requested
        heatmap_b64 = None
        if request.form.get('generate_heatmap', 'false').lower() == 'true':
            try:
                heatmap_b64 = viz_generator.generate_gradcam_heatmap(
                    model_manager.model, processed_image, np.array(image)
                )
            except Exception as e:
                logger.warning(f"Failed to generate heatmap: {str(e)}")
        
        # Format response
        response = {
            'disease': prediction_results['predicted_class'],
            'confidence': float(prediction_results['confidence']),
            'all_predictions': prediction_results['all_predictions'],
            'processing_time': prediction_results['processing_time'],
            'image_quality': _assess_image_quality(np.array(image)),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': model_manager.get_model_info()['name'],
                'version': model_manager.get_model_info()['version']
            }
        }
        
        if heatmap_b64:
            response['heatmap'] = heatmap_b64
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Endpoint to retrain the model with new data
    This should be secured in production
    """
    try:
        data_path = request.json.get('data_path')
        if not data_path or not os.path.exists(data_path):
            return jsonify({'error': 'Invalid data path provided'}), 400
        
        # Start training (this could be made async in production)
        training_results = model_manager.train_model(data_path)
        
        return jsonify({
            'message': 'Model training completed',
            'results': training_results
        })
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': 'Internal server error during training'}), 500

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    return jsonify(model_manager.get_model_info())

def _allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _assess_image_quality(image):
    """Assess image quality based on various metrics"""
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate image quality metrics
        height, width = gray.shape
        
        # Check resolution
        if width < 224 or height < 224:
            return "Poor - Low Resolution"
        
        # Calculate variance of Laplacian (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var > 100:
            return "Excellent"
        elif laplacian_var > 50:
            return "Good"
        elif laplacian_var > 20:
            return "Fair"
        else:
            return "Poor - Blurry Image"
    
    except Exception as e:
        logger.warning(f"Image quality assessment failed: {str(e)}")
        return "Unknown"

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(
        host=app.config.get('HOST', '127.0.0.1'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', False)
    )
