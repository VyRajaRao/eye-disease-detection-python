#!/usr/bin/env python3
"""
Performance testing script for EyeZen Detect API
Tests both frontend and backend loading times
"""

import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import sys

class PerformanceTester:
    def __init__(self):
        self.backend_url = "http://localhost:5000"
        self.frontend_url = "http://localhost:5173"
        
    def test_backend_health(self):
        """Test backend health endpoint performance"""
        print("🔧 Testing backend health endpoint...")
        times = []
        
        for i in range(5):
            start = time.time()
            try:
                response = requests.get(f"{self.backend_url}/api/health", timeout=10)
                end = time.time()
                
                if response.status_code == 200:
                    times.append((end - start) * 1000)  # Convert to ms
                    print(f"  Health check {i+1}: {times[-1]:.2f}ms")
                else:
                    print(f"  Health check {i+1}: Failed (HTTP {response.status_code})")
            except Exception as e:
                print(f"  Health check {i+1}: Error - {str(e)}")
        
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"📊 Backend Health Stats:")
            print(f"   Average: {avg_time:.2f}ms")
            print(f"   Min: {min_time:.2f}ms") 
            print(f"   Max: {max_time:.2f}ms")
            
            if avg_time < 100:
                print("✅ Backend response time is EXCELLENT (<100ms)")
            elif avg_time < 500:
                print("⚠️ Backend response time is GOOD (<500ms)")
            else:
                print("❌ Backend response time is SLOW (>500ms)")
        else:
            print("❌ Backend is not responding!")
        
        return times
    
    def test_frontend_loading(self):
        """Test frontend loading performance"""
        print("\n🎨 Testing frontend loading...")
        
        try:
            start = time.time()
            response = requests.get(self.frontend_url, timeout=10)
            end = time.time()
            
            load_time = (end - start) * 1000
            print(f"  Frontend load time: {load_time:.2f}ms")
            
            if load_time < 1000:
                print("✅ Frontend load time is EXCELLENT (<1s)")
            elif load_time < 3000:
                print("⚠️ Frontend load time is GOOD (<3s)")
            else:
                print("❌ Frontend load time is SLOW (>3s)")
                
            return load_time
            
        except Exception as e:
            print(f"❌ Frontend not accessible: {str(e)}")
            return None
    
    def test_concurrent_requests(self, num_requests=10):
        """Test concurrent request handling"""
        print(f"\n🚀 Testing concurrent requests ({num_requests} requests)...")
        
        def make_request():
            start = time.time()
            try:
                response = requests.get(f"{self.backend_url}/api/health", timeout=10)
                end = time.time()
                return (end - start) * 1000, response.status_code == 200
            except:
                return None, False
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = []
            successful = 0
            
            for future in as_completed(futures):
                result = future.result()
                if result[0] is not None:
                    results.append(result[0])
                    if result[1]:
                        successful += 1
        
        if results:
            avg_time = statistics.mean(results)
            print(f"📊 Concurrent Request Stats:")
            print(f"   Successful: {successful}/{num_requests}")
            print(f"   Average response time: {avg_time:.2f}ms")
            print(f"   Throughput: {successful/max(results)/1000:.2f} req/s" if results else "N/A")
            
            if successful == num_requests and avg_time < 200:
                print("✅ Concurrent handling is EXCELLENT")
            elif successful >= num_requests * 0.8:
                print("⚠️ Concurrent handling is GOOD")
            else:
                print("❌ Concurrent handling needs improvement")
        
        return results
    
    def test_model_loading_time(self):
        """Test model loading simulation"""
        print("\n🧠 Testing model availability...")
        
        try:
            response = requests.get(f"{self.backend_url}/api/model/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Model is loaded and ready")
                return True
            else:
                print("❌ Model endpoint not accessible")
                return False
        except Exception as e:
            print(f"❌ Model test failed: {str(e)}")
            return False
    
    def run_full_performance_test(self):
        """Run comprehensive performance test"""
        print("🎯 EyeZen Detect Performance Test")
        print("=" * 50)
        
        # Test backend
        backend_times = self.test_backend_health()
        
        # Test frontend
        frontend_time = self.test_frontend_loading()
        
        # Test concurrent handling
        concurrent_times = self.test_concurrent_requests()
        
        # Test model
        model_ready = self.test_model_loading_time()
        
        # Summary
        print("\n📋 Performance Summary")
        print("=" * 30)
        
        if backend_times:
            avg_backend = statistics.mean(backend_times)
            print(f"Backend Average: {avg_backend:.2f}ms")
        else:
            print("Backend: ❌ Not responding")
        
        if frontend_time:
            print(f"Frontend Load: {frontend_time:.2f}ms")
        else:
            print("Frontend: ❌ Not accessible")
        
        if concurrent_times:
            avg_concurrent = statistics.mean(concurrent_times)
            print(f"Concurrent Avg: {avg_concurrent:.2f}ms")
        
        print(f"Model Ready: {'✅ Yes' if model_ready else '❌ No'}")
        
        # Overall assessment
        print("\n🏆 Overall Assessment:")
        if (backend_times and statistics.mean(backend_times) < 200 and 
            frontend_time and frontend_time < 2000 and model_ready):
            print("✅ Application performance is EXCELLENT!")
        elif (backend_times and statistics.mean(backend_times) < 500 and
              frontend_time and frontend_time < 5000):
            print("⚠️ Application performance is GOOD with room for improvement")
        else:
            print("❌ Application performance needs optimization")

if __name__ == "__main__":
    tester = PerformanceTester()
    
    try:
        tester.run_full_performance_test()
    except KeyboardInterrupt:
        print("\n\n⏹️ Performance test interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance test failed: {str(e)}")
    
    print("\n📝 Recommendations:")
    print("1. Ensure both frontend and backend servers are running")
    print("2. Check network connectivity")
    print("3. Monitor system resources (CPU, Memory)")
    print("4. Consider implementing CDN for static assets")
    print("5. Use Redis for caching in production")
