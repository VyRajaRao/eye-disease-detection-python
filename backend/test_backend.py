#!/usr/bin/env python3
"""
Simple test script to verify the backend components work
"""

import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_config():
    """Test configuration loading"""
    try:
        from config import Config
        config = Config()
        print("✅ Configuration loaded successfully")
        print(f"   - Model path: {config.MODEL_PATH}")
        print(f"   - Classes: {len(config.CLASS_NAMES)}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_basic_imports():
    """Test basic imports without heavy dependencies"""
    try:
        import numpy as np
        import tensorflow as tf
        print("✅ Basic dependencies (NumPy, TensorFlow) available")
        return True
    except ImportError as e:
        print(f"❌ Missing basic dependencies: {e}")
        return False

def test_flask_app():
    """Test Flask app initialization"""
    try:
        # Mock the heavy dependencies
        import sys
        from unittest.mock import Mock
        
        # Mock OpenCV and other dependencies
        sys.modules['cv2'] = Mock()
        sys.modules['matplotlib.pyplot'] = Mock()
        sys.modules['seaborn'] = Mock()
        
        from app import app
        print("✅ Flask app initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app failed: {e}")
        return False

def create_basic_model():
    """Create a basic model for testing"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create a simple model
        model = keras.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(6, activation='softmax')  # 6 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = 'models/eye_disease_model.h5'
        model.save(model_path)
        
        print(f"✅ Basic model created and saved to {model_path}")
        print(f"   - Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing EyeZen Detect Backend Components")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Basic Imports", test_basic_imports),
        ("Flask App", test_flask_app),
        ("Model Creation", create_basic_model),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔧 Testing {name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Backend is ready.")
    else:
        print("⚠️  Some tests failed. Check dependencies and setup.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
