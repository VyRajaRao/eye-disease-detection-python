#!/usr/bin/env python3
"""
Quick API Test Script
Tests the EyeZen Detect API endpoints
"""

import requests
import time
import json
from pathlib import Path

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    try:
        print("🧠 Testing model info endpoint...")
        response = requests.get('http://localhost:5000/api/model/info', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved!")
            print(f"   Name: {data.get('name')}")
            print(f"   Classes: {data.get('num_classes')}")
            print(f"   Trained: {data.get('trained')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint with a sample image"""
    try:
        print("🔍 Testing prediction endpoint...")
        
        # Find a sample image from our dataset
        sample_image = None
        dataset_path = Path("dataset")
        
        for condition_folder in dataset_path.iterdir():
            if condition_folder.is_dir():
                for image_file in condition_folder.glob("*.jpg"):
                    sample_image = image_file
                    break
                if sample_image:
                    break
        
        if not sample_image:
            print("❌ No sample image found in dataset")
            return False
        
        print(f"   Using sample image: {sample_image}")
        
        # Prepare the request
        files = {'image': open(sample_image, 'rb')}
        data = {'generate_heatmap': 'false'}  # Skip heatmap for quick test
        
        response = requests.post('http://localhost:5000/api/predict', 
                               files=files, data=data, timeout=30)
        
        files['image'].close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Predicted disease: {result.get('disease')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"   Image quality: {result.get('image_quality')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 " + "="*50)
    print("   EyeZen Detect API Test Suite")
    print("🧪 " + "="*50 + "\n")
    
    print("⚠️  Make sure the backend server is running on http://localhost:5000")
    print("   You can start it with: backend\\venv\\Scripts\\python backend\\app.py\n")
    
    # Wait a moment for user to start server if needed
    print("⏳ Starting tests in 3 seconds...")
    time.sleep(3)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Model Info", test_model_info_endpoint),
        ("Prediction", test_prediction_endpoint),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Start the frontend: npm run dev")
        print("2. Open http://localhost:5173 in your browser")
        print("3. Upload an eye image to test the full application")
    else:
        print("⚠️  Some tests failed. Check the backend server status.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
