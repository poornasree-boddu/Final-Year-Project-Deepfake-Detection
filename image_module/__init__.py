"""
Image Deepfake Detection Module

EfficientNet-B0 based image deepfake detector
"""

from .inference import predict_image
from .gradcam import generate_image_gradcam
from .model import ImageDeepfakeDetector

__all__ = ['predict_image', 'generate_image_gradcam', 'ImageDeepfakeDetector']
