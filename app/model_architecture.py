"""
Brain Tumor CNN Model Architecture
Enhanced 2D+1D CNN with Transfer Learning and Attention
"""
import sys
from pathlib import Path

# Import the actual model
from models.cnn import ChannelAttention, Enhanced_CNN2D1D, SpatialAttention

# Add src to path so we can import the model
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Export for use in other modules
BrainTumorCNN = Enhanced_CNN2D1D

__all__ = ["BrainTumorCNN", "Enhanced_CNN2D1D", "ChannelAttention", "SpatialAttention"]
