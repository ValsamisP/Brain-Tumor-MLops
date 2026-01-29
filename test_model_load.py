"""
Test script to verify model loads correctly
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.cnn import Enhanced_CNN2D1D

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

# Test 1: Model instantiation
print("\n1. Testing model instantiation...")
try:
    model = Enhanced_CNN2D1D(num_classes=4, pretrained=False)
    print("Model instantiated successfully")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"Model instantiation failed: {e}")
    sys.exit(1)

# Test 2: Load checkpoint
print("\n2. Testing checkpoint loading...")
model_path = "models/best_model.pth"
try:
    checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
    print(f"Checkpoint loaded successfully")
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
except Exception as e:
    print(f"Checkpoint loading failed: {e}")
    sys.exit(1)

# Test 3: Load weights into model
print("\n3. Testing weight loading...")
try:
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Weights loaded from 'model_state_dict' key")
    else:
        model.load_state_dict(checkpoint)
        print("Weights loaded directly")
except Exception as e:
    print(f"Weight loading failed: {e}")
    sys.exit(1)

# Test 4: Model info
print("\n4. Model information:")
if 'class_names' in checkpoint:
    print(f"   Classes: {checkpoint['class_names']}")
if 'best_val_acc' in checkpoint:
    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")

# Test 5: Test forward pass
print("\n5. Testing forward pass...")
try:
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Forward pass successful")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output (logits): {output}")
except Exception as e:
    print(f"Forward pass failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nMy model is ready for deployment!")