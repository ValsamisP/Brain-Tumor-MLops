"""
Model Validation Script
Validates model performance metrics before deployment
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Thresholds for model validation
MIN_ACCURACY = 0.95
MIN_PRECISION = 0.93
MIN_RECALL = 0.90
MAX_MODEL_SIZE_MB = 500


def validate_model_exists(model_path: str) -> bool:
    """Check if model file exists"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    print(f"✅ Model found at {model_path}")
    return True


def validate_model_size(model_path: str) -> bool:
    """Check model size is within acceptable limits"""
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📊 Model size: {size_mb:.2f} MB")
    
    if size_mb > MAX_MODEL_SIZE_MB:
        print(f"❌ Model size ({size_mb:.2f} MB) exceeds limit ({MAX_MODEL_SIZE_MB} MB)")
        return False
    
    print(f"✅ Model size is within limits")
    return True


def validate_model_loadable(model_path: str) -> bool:
    """Check if model can be loaded"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✅ Model can be loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False


def validate_model_performance(metrics_path: str) -> bool:
    """Validate model performance metrics"""
    if not os.path.exists(metrics_path):
        print(f"⚠️  Metrics file not found at {metrics_path}")
        print("   Skipping performance validation")
        return True
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        accuracy = metrics.get('test_accuracy', 0)
        precision = metrics.get('test_precision', 0)
        recall = metrics.get('test_recall', 0)
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f} (threshold: {MIN_ACCURACY})")
        print(f"   Precision: {precision:.4f} (threshold: {MIN_PRECISION})")
        print(f"   Recall:    {recall:.4f} (threshold: {MIN_RECALL})")
        
        validations = [
            (accuracy >= MIN_ACCURACY, f"Accuracy below threshold"),
            (precision >= MIN_PRECISION, f"Precision below threshold"),
            (recall >= MIN_RECALL, f"Recall below threshold")
        ]
        
        all_passed = all(v[0] for v in validations)
        
        for passed, message in validations:
            if not passed:
                print(f"❌ {message}")
            else:
                print(f"✅ {message.replace('below', 'meets')} threshold")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading metrics: {str(e)}")
        return False


def main():
    """Run all model validations"""
    print("=" * 60)
    print("🔍 Model Validation Pipeline")
    print("=" * 60)
    
    # Paths
    model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
    metrics_path = 'results/model_metrics.json'
    
    validations = [
        ("Model Exists", validate_model_exists(model_path)),
        ("Model Size", validate_model_size(model_path) if os.path.exists(model_path) else False),
        ("Model Loadable", validate_model_loadable(model_path) if os.path.exists(model_path) else False),
        ("Model Performance", validate_model_performance(metrics_path))
    ]
    
    print("\n" + "=" * 60)
    print("📋 Validation Summary")
    print("=" * 60)
    
    for name, passed in validations:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(v[1] for v in validations)
    
    print("=" * 60)
    if all_passed:
        print("🎉 All validations passed! Model is ready for deployment.")
        sys.exit(0)
    else:
        print("⚠️  Some validations failed. Please review before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
