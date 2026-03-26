"""
Image Deepfake Detection - Inference Pipeline

Provides the main inference function for image deepfake detection.
"""

import torch
from PIL import Image
from torchvision import transforms
from typing import Dict
import os
import sys

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_MODEL_PATH

from image_module.model import ImageDeepfakeDetector
from image_module.gradcam import generate_image_gradcam


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model (singleton pattern)
_model = None
_transform = None


def load_model():
    """Load the trained model (singleton pattern)"""
    global _model, _transform
    
    if _model is None:
        # Create model
        _model = ImageDeepfakeDetector(pretrained=False)
        
        # Load trained weights
        _model.load_state_dict(
            torch.load(IMAGE_MODEL_PATH, map_location=device, weights_only=True)
        )
        _model = _model.to(device)
        _model.eval()
        
        # Create transform
        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return _model, _transform


def predict_image(image_path: str, generate_gradcam: bool = True) -> Dict:
    """
    Main inference function for image deepfake detection
    
    Args:
        image_path: Path to image file
        generate_gradcam: Whether to return GradCAM overlay data
        
    Returns:
        Dictionary containing:
            - prediction: "Real" or "Fake"
            - confidence: Confidence score (0-100)
            - probabilities: Dict with 'real' and 'fake' probabilities
    """
    try:
        # Load model
        model, transform = load_model()
        
        # Load and preprocess image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        print("Running inference...")
        with torch.no_grad():
            output = model(image_tensor)
            prob_fake = torch.sigmoid(output).item()
        
        prob_real = 1 - prob_fake
        
        # Determine prediction
        if prob_fake > 0.5:
            prediction = "Fake"
            confidence = prob_fake * 100
        else:
            prediction = "Real"
            confidence = prob_real * 100
        
        # Prepare result
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(prob_real * 100, 2),
                "fake": round(prob_fake * 100, 2)
            }
        }

        if generate_gradcam:
            original_rgb, gradcam_rgb = generate_image_gradcam(image_path)
            result["original"] = original_rgb
            result["gradcam"] = gradcam_rgb
        
        print(f"Prediction: {prediction} (Confidence: {confidence:.2f}%)")
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    """Test the inference pipeline"""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets", "image_dataset", "Test", "Fake", "easy_0_1.jpg"
        )
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("\nUsage: python inference.py <path_to_image>")
        sys.exit(1)
    
    print("=" * 60)
    print("IMAGE DEEPFAKE DETECTION - INFERENCE TEST")
    print("=" * 60)
    print(f"\nImage: {image_path}")
    print(f"Device: {device}")
    print(f"Model: {IMAGE_MODEL_PATH}")
    print("-" * 60)
    
    # Run prediction
    result = predict_image(image_path, generate_gradcam=True)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Probabilities:")
    print(f"  - Real: {result['probabilities']['real']}%")
    print(f"  - Fake: {result['probabilities']['fake']}%")
    print("=" * 60)
