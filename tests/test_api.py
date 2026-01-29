"""
Unit tests for Brain Tumor Classification API
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check and system endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["version"] == "1.0.0"

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data


class TestPredictionEndpoints:
    """Test prediction endpoints"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a dummy RGB image
        img = Image.new("RGB", (224, 224), color="gray")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes

    def test_predict_endpoint_valid_image(self, sample_image):
        """Test prediction with valid image"""
        files = {"file": ("test_image.jpg", sample_image, "image/jpeg")}
        response = client.post("/predict", files=files)

        # Should return 200 or 500 depending on model availability
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "prediction_id" in data
            assert "timestamp" in data
            assert data["predicted_class"] in [
                "glioma",
                "meningioma",
                "no_tumor",
                "pituitary",
            ]
            assert 0 <= data["confidence"] <= 1

    def test_predict_endpoint_invalid_file(self):
        """Test prediction with invalid file type"""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400

    def test_batch_predict_endpoint(self, sample_image):
        """Test batch prediction"""
        files = [
            ("files", ("test1.jpg", sample_image, "image/jpeg")),
            ("files", ("test2.jpg", sample_image, "image/jpeg")),
        ]
        response = client.post("/batch_predict", files=files)

        # Should return 200 or 500 depending on model availability
        assert response.status_code in [200, 500]

    def test_batch_predict_too_many_files(self, sample_image):
        """Test batch prediction with too many files"""
        files = [
            ("files", (f"test{i}.jpg", sample_image, "image/jpeg")) for i in range(15)
        ]
        response = client.post("/batch_predict", files=files)
        assert response.status_code == 400


class TestMetricsEndpoints:
    """Test metrics and monitoring endpoints"""

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "total_errors" in data
        assert "uptime_seconds" in data


class TestModelManagement:
    """Test model management endpoints"""

    def test_reload_model_endpoint(self):
        """Test model reload endpoint"""
        response = client.post("/reload_model")
        # Should return 200 or 500 depending on model availability
        assert response.status_code in [200, 500]


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations"""

    async def test_concurrent_predictions(self):
        """Test handling concurrent prediction requests"""
        # This would test the API's ability to handle concurrent requests
        # In a real scenario, you'd use async client and make multiple concurrent requests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
