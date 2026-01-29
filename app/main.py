"""
Brain Tumor Classification API
FastAPI application for serving the trained CNN model
"""
import io
import logging
from typing import Dict, List
from datetime import datetime

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torchvision.transforms as transforms

from .model_loader import ModelLoader
from .monitoring import MetricsCollector, log_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Classification API",
    description="Deep Learning API for classifying brain MRI scans",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add root route to serve the frontend
@app.get("/", include_in_schema=False)
async def serve_frontend(request: Request):
    """Serve the frontend UI"""
    return templates.TemplateResponse("index.html", {"request": request})

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader and metrics
model_loader = ModelLoader()
metrics_collector = MetricsCollector()

# Class labels
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    prediction_id: str
    timestamp: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Brain Tumor Classification API...")
    try:
        model_loader.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Brain Tumor Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "model_version": model_loader.get_version(),
        "uptime_seconds": model_loader.get_uptime()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict tumor type from MRI image
    
    Args:
        file: Uploaded MRI image file (jpg, png, jpeg)
        
    Returns:
        Prediction results with class, confidence, and probabilities
    """
    start_time = datetime.now()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, jpeg)"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        prediction = model_loader.predict(image)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Log prediction for monitoring
        log_prediction(
            prediction_id=prediction_id,
            predicted_class=prediction['class'],
            confidence=prediction['confidence'],
            processing_time_ms=processing_time
        )
        
        # Update metrics
        metrics_collector.record_prediction(
            prediction['class'],
            prediction['confidence'],
            processing_time
        )
        
        return {
            "predicted_class": prediction['class'],
            "confidence": prediction['confidence'],
            "probabilities": prediction['probabilities'],
            "prediction_id": prediction_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        metrics_collector.record_error()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[PredictionResponse])
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple MRI images
    
    Args:
        files: List of uploaded MRI image files
        
    Returns:
        List of prediction results
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {str(e)}")
            continue
    
    return results


@app.get("/metrics")
async def get_metrics():
    """Get current metrics and statistics"""
    return metrics_collector.get_metrics()


@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for model updates)"""
    try:
        model_loader.reload_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
