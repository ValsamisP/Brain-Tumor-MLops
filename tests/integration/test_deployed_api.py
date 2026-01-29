"""
Integration tests for deployed Brain Tumor Classification API
Tests the API in a real deployment environment
"""
import io
import time

import pytest
import requests
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class TestDeployedAPI:
    """Test the deployed API"""

    @classmethod
    def setup_class(cls):
        """Wait for API to be ready"""
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    print("API is ready")
                    return
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    print(f"Waiting for API... (attempt {i+1}/{max_retries})")
                    time.sleep(3)
                else:
                    raise Exception("API did not become ready in time")

    def create_test_image(self):
        """Create a test image"""
        img = Image.new("RGB", (224, 224), color="gray")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_predict_endpoint(self):
        """Test prediction endpoint"""
        img_bytes = self.create_test_image()
        files = {"file": ("test_brain_scan.jpg", img_bytes, "image/jpeg")}

        response = requests.post(
            f"{API_BASE_URL}/predict", files=files, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "prediction_id" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data

        # Validate data types and ranges
        assert isinstance(data["predicted_class"], str)
        assert 0 <= data["confidence"] <= 1
        assert len(data["probabilities"]) == 4
        assert sum(data["probabilities"].values()) > 0.99  # Should sum to ~1

    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        files = [
            ("files", ("test1.jpg", self.create_test_image(), "image/jpeg")),
            ("files", ("test2.jpg", self.create_test_image(), "image/jpeg")),
            ("files", ("test3.jpg", self.create_test_image(), "image/jpeg")),
        ]

        response = requests.post(
            f"{API_BASE_URL}/batch_predict", files=files, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()

        assert "total_predictions" in data
        assert "average_processing_time_ms" in data
        assert "class_distribution" in data

    def test_invalid_file_type(self):
        """Test with invalid file type"""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = requests.post(
            f"{API_BASE_URL}/predict", files=files, timeout=TIMEOUT
        )
        assert response.status_code == 400

    def test_performance_benchmark(self):
        """Benchmark prediction performance"""
        num_requests = 10
        times = []

        for _ in range(num_requests):
            img_bytes = self.create_test_image()
            files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict", files=files, timeout=TIMEOUT
            )
            end = time.time()

            if response.status_code == 200:
                times.append(end - start)

        if times:
            avg_time = sum(times) / len(times)
            print(f"\nAverage prediction time: {avg_time:.3f}s")
            print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")

            # Assert reasonable performance (adjust threshold as needed)
            assert avg_time < 5.0, "Average prediction time too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
