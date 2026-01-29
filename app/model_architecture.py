"""
Brain Tumor CNN Model Architecture
Enhanced 2D+1D CNN with Transfer Learning and Attention
"""
import os
import sys

# Now import from src.models.cnn
from src.models.cnn import ChannelAttention, Enhanced_CNN2D1D, SpatialAttention

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Export for use in other modules
BrainTumorCNN = Enhanced_CNN2D1D

__all__ = ["BrainTumorCNN", "Enhanced_CNN2D1D", "ChannelAttention", "SpatialAttention"]
