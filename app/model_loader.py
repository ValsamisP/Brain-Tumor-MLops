"""
Model Loader Module
Handles model loading, caching, and inference
"""
import logging
import os
import time
from typing import Dict

import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading and inference"""

    def __init__(self, model_path: str = None):
        # Use flexible path that works locally and in Docker
        if model_path:
            self.model_path = model_path
        else:
            # Try env variable first
            self.model_path = os.getenv("MODEL_PATH")

            # If not set relative path (this is working locally)
            if not self.model_path:
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.model_path = os.path.join(base_path, "models", "best_model.pth")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
        self.version = "1.0.0"
        self.load_time = None

        # Image preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"Model loader initialized with device: {self.device}")

    def load_model(self):
        """Load the trained model"""
        try:
            start_time = time.time()

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Load model checkpoint
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )

            # Initialize model architecture
            from .model_architecture import BrainTumorCNN

            self.model = BrainTumorCNN(
                num_classes=len(self.class_names), pretrained=False
            )

            # Load state dict
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model from checkpoint with model_state_dict_key")

                # Extract additional info if available
                if "class_names" in checkpoint:
                    logger.info(f"Checkpoint classes: {checkpoint['class_names']}")
                if "best_val_acc" in checkpoint:
                    logger.info(
                        f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}"
                    )
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model from checkpoint (direct state_dict)")

            self.model.to(self.device)
            self.model.eval()

            self.load_time = time.time()
            load_duration = time.time() - start_time

            logger.info(f"Model loaded successfully in {load_duration:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, image: Image.Image) -> Dict:
        """
        Make prediction on a single image

        Args:
            image: PIL Image object

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Get class name and probabilities
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()

            # Create probability dictionary
            probs_dict = {
                self.class_names[i]: float(probabilities[0][i])
                for i in range(len(self.class_names))
            }

            return {
                "class": predicted_class,
                "confidence": confidence_score,
                "probabilities": probs_dict,
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def reload_model(self):
        """Reload the model"""
        logger.info("Reloading model...")
        self.model = None
        self.load_model()

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_version(self) -> str:
        """Get model version"""
        return self.version

    def get_uptime(self) -> float:
        """Get uptime in seconds since model load"""
        if self.load_time:
            return time.time() - self.load_time
        return 0.0
