#!/usr/bin/env python3
"""
Heatmap Integration Test for EyeZen Detect
Tests backend heatmap generation and frontend integration
"""

import os
import sys
import requests
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
import io

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def create_test_image():
    """Create a synthetic retinal image for testing"""
    # Create a 224x224 RGB image that looks like a retinal scan
    img = Image.new('RGB', (224, 224), color=(20, 10, 10))
    draw = ImageDraw.Draw(img)
    
    # Draw a circular retinal pattern
    draw.ellipse([20, 20, 204, 204], fill=(80, 40, 30), outline=(120, 60, 40))
    draw.ellipse([50, 50, 174, 174], fill=(60, 30, 20), outline=(100, 50, 30))
    
    # Add some vessel-like patterns
    draw.line([50, 112, 174, 112], fill=(40, 20, 15), width=3)
    draw.line([112, 50, 112, 174], fill=(40, 20, 15), width=2)
    draw.line([80, 80, 144, 144], fill=(50, 25, 20), width=2)
    
    # Add optic disc simulation
    draw.ellipse([95, 95, 129, 129], fill=(150, 120, 100), outline=(180, 150, 130))
    
    return img

def test_backend_heatmap_generation():
    """Test backend heatmap generation functionality"""
    print("🧠 Testing Backend Heatmap Generation...")
    
    try:
        from utils.visualization import VisualizationGenerator
        from utils.model_manager import ModelManager
        from utils.image_preprocessor import ImagePreprocessor
        
        # Initialize components
        viz_gen = VisualizationGenerator()
        model_mgr = ModelManager()
        img_proc = ImagePreprocessor()
        
        # Create test image
        test_img = create_test_image()
        test_img_array = np.array(test_img)
        
        print("✅ Created synthetic test image")
        
        # Preprocess image
        processed_img = img_proc.preprocess_for_prediction(test_img_array)
        print("✅ Preprocessed test image")
        
        # Try to load model (will create basic model if none exists)
        model_loaded = model_mgr.load_model()
        if model_loaded:
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Using basic model (no trained model found)")
        
        # Test heatmap generation
        heatmap_b64 = viz_gen.generate_gradcam_heatmap(
            model_mgr.model, 
            processed_img, 
            test_img_array
        )
        
        if heatmap_b64:
            print("✅ Heatmap generated successfully")
            print(f"   Heatmap size: {len(heatmap_b64)} characters")
            
            # Validate base64 encoding
            try:
                heatmap_data = base64.b64decode(heatmap_b64)
                print(f"   Decoded size: {len(heatmap_data)} bytes")
                return True
            except Exception as e:
                print(f"❌ Invalid base64 encoding: {e}")
                return False
        else:
            print("❌ Heatmap generation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Backend heatmap test failed: {e}")
        return False

def test_api_heatmap_endpoint():
    """Test API heatmap generation endpoint"""
    print("\n🔧 Testing API Heatmap Endpoint...")
    
    backend_url = "http://localhost:5000"
    
    # Check if backend is running
    try:
        response = requests.get(f"{backend_url}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not running or unhealthy")
            return False
        print("✅ Backend is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to backend")
        return False
    
    # Create test image file
    test_img = create_test_image()
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Test prediction with heatmap
    try:
        files = {'image': ('test_retina.png', img_bytes, 'image/png')}
        data = {'generate_heatmap': 'true'}
        
        response = requests.post(
            f"{backend_url}/api/predict", 
            files=files, 
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API prediction successful")
            print(f"   Disease: {result.get('disease', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            
            if 'heatmap' in result and result['heatmap']:
                print("✅ Heatmap included in response")
                print(f"   Heatmap size: {len(result['heatmap'])} characters")
                
                # Test heatmap validity
                try:
                    heatmap_data = base64.b64decode(result['heatmap'])
                    print(f"   Decoded size: {len(heatmap_data)} bytes")
                    return True
                except Exception as e:
                    print(f"❌ Invalid heatmap data: {e}")
                    return False
            else:
                print("❌ No heatmap in response")
                return False
        else:
            print(f"❌ API request failed with status {response.status_code}")
            if response.headers.get('content-type', '').startswith('application/json'):
                error = response.json()
                print(f"   Error: {error.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False

def test_frontend_heatmap_display():
    """Test frontend heatmap display functionality"""
    print("\n🎨 Testing Frontend Heatmap Display...")
    
    frontend_url = "http://localhost:8080"
    
    # Check if frontend is running
    try:
        response = requests.get(frontend_url, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            
            # Check if the ResultsSection component has heatmap functionality
            content = response.text
            if 'View AI Heatmap' in content or 'heatmap' in content.lower():
                print("✅ Frontend has heatmap display functionality")
                return True
            else:
                print("⚠️ Frontend may not have heatmap display (checking source)")
                
                # Check source files for heatmap integration
                try:
                    results_component_path = os.path.join(
                        os.path.dirname(__file__), 
                        'src', 'components', 'ResultsSection.tsx'
                    )
                    if os.path.exists(results_component_path):
                        with open(results_component_path, 'r') as f:
                            component_code = f.read()
                            if 'heatmap' in component_code.lower() and 'View AI Heatmap' in component_code:
                                print("✅ Heatmap functionality found in ResultsSection component")
                                return True
                            else:
                                print("❌ Heatmap functionality not found in ResultsSection component")
                                return False
                    else:
                        print("❌ ResultsSection component not found")
                        return False
                except Exception as e:
                    print(f"❌ Error checking component: {e}")
                    return False
        else:
            print(f"❌ Frontend not accessible (HTTP {response.status_code})")
            return False
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to frontend")
        return False

def run_comprehensive_heatmap_test():
    """Run comprehensive heatmap integration test"""
    print("🔥 EyeZen Detect - Heatmap Integration Test")
    print("=" * 50)
    
    tests = [
        ("Backend Heatmap Generation", test_backend_heatmap_generation),
        ("API Heatmap Endpoint", test_api_heatmap_endpoint),
        ("Frontend Heatmap Display", test_frontend_heatmap_display)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📝 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Heatmap Integration Test Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🏆 Overall Result: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All heatmap functionality is working correctly!")
        print("\n✨ Heatmap Features Available:")
        print("• Backend Grad-CAM heatmap generation")
        print("• API endpoint for heatmap requests") 
        print("• Frontend heatmap visualization")
        print("• Interactive heatmap viewing")
    elif passed >= len(tests) * 0.7:
        print("⚠️ Most heatmap functionality is working with minor issues")
    else:
        print("❌ Heatmap integration needs significant fixes")
    
    print("\n📝 Troubleshooting Tips:")
    if not results.get("Backend Heatmap Generation", False):
        print("• Install missing Python dependencies (tensorflow, matplotlib, opencv)")
        print("• Check model file exists or can be created")
        print("• Verify visualization utility imports")
    
    if not results.get("API Heatmap Endpoint", False):
        print("• Ensure backend server is running on port 5000")
        print("• Check API routes and CORS configuration")
        print("• Verify Flask dependencies")
    
    if not results.get("Frontend Heatmap Display", False):
        print("• Ensure frontend server is running on port 8080")
        print("• Check ResultsSection component has heatmap code")
        print("• Verify React components are properly integrated")
    
    return passed == len(tests)

if __name__ == "__main__":
    try:
        success = run_comprehensive_heatmap_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Heatmap integration test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        sys.exit(1)
