"""
Smoke Tests
Quick validation tests after deployment
"""
import requests
import time
import sys
from typing import Dict, Tuple


API_URL = "https://api.yourdomain.com"  # Update with your actual URL
TIMEOUT = 10


def test_health_check() -> Tuple[bool, str]:
    """Test if API is responding"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                return True, "✅ Health check passed"
            return False, f"❌ API unhealthy: {data}"
        return False, f"❌ Health check failed with status {response.status_code}"
    except Exception as e:
        return False, f"❌ Health check error: {str(e)}"


def test_model_loaded() -> Tuple[bool, str]:
    """Test if model is loaded"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get("model_loaded"):
                return True, "✅ Model is loaded"
            return False, "❌ Model is not loaded"
        return False, "❌ Could not verify model status"
    except Exception as e:
        return False, f"❌ Model check error: {str(e)}"


def test_prediction_endpoint() -> Tuple[bool, str]:
    """Test basic prediction functionality"""
    try:
        # Create a simple test image
        from PIL import Image
        import io
        
        img = Image.new('RGB', (224, 224), color='gray')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["predicted_class", "confidence", "probabilities"]
            if all(field in data for field in required_fields):
                return True, "✅ Prediction endpoint working"
            return False, f"❌ Missing fields in response: {data.keys()}"
        return False, f"❌ Prediction failed with status {response.status_code}"
    except Exception as e:
        return False, f"❌ Prediction test error: {str(e)}"


def test_response_time() -> Tuple[bool, str]:
    """Test if response time is acceptable"""
    try:
        start = time.time()
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        elapsed = time.time() - start
        
        if elapsed < 1.0:  # Should respond within 1 second
            return True, f"✅ Response time OK ({elapsed:.3f}s)"
        return False, f"⚠️  Slow response time ({elapsed:.3f}s)"
    except Exception as e:
        return False, f"❌ Response time test error: {str(e)}"


def run_smoke_tests():
    """Run all smoke tests"""
    print("=" * 60)
    print(" Running Smoke Tests")
    print(f"   Target: {API_URL}")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Loaded", test_model_loaded),
        ("Prediction Endpoint", test_prediction_endpoint),
        ("Response Time", test_response_time)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        passed, message = test_func()
        results.append((test_name, passed, message))
        print(f"   {message}")
        time.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "=" * 60)
    print(" Smoke Test Results")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, message in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All smoke tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some smoke tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_tests()
