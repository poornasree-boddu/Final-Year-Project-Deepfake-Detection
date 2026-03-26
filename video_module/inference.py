"""
Video Deepfake Detection - Inference Pipeline

This module provides the main inference function for video deepfake detection.
Uses GoogLeNet + BiLSTM architecture with GradCAM visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from torchvision.models import GoogLeNet_Weights
from PIL import Image
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VIDEO_MODEL_PATH, FRAME_COUNT, FEATURE_DIM, LSTM_HIDDEN_SIZE

# ---------- DEVICE CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LSTM MODEL ----------
class LSTMModel(nn.Module):
    """BiLSTM model for temporal deepfake detection"""
    def __init__(self, input_size=1024, hidden_size=256, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

# ---------- GLOBAL MODELS ----------
_lstm_model = None
_googlenet_model = None
_transform = None

def load_models():
    """Load LSTM and GoogLeNet models (singleton pattern)"""
    global _lstm_model, _googlenet_model, _transform
    
    if _lstm_model is None:
        # Load LSTM model
        _lstm_model = LSTMModel(
            input_size=FEATURE_DIM,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=1
        ).to(device)
        _lstm_model.load_state_dict(
            torch.load(VIDEO_MODEL_PATH, map_location=device, weights_only=True)
        )
        _lstm_model.eval()

        # Load GoogLeNet for feature extraction
        _googlenet_model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT)
        _googlenet_model.fc = torch.nn.Identity()
        _googlenet_model = _googlenet_model.to(device)
        _googlenet_model.eval()

        # Image transform
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    return _lstm_model, _googlenet_model, _transform

# ---------- FRAME EXTRACTION ----------
def extract_frames_from_video(video_path: str, num_frames: int = FRAME_COUNT) -> List[np.ndarray]:
    """
    Extract evenly spaced frames from video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    frames = []
    count = 0
    saved = 0
    
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frames.append(frame)
            saved += 1
        
        count += 1
    
    cap.release()
    
    # Pad if we didn't get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    return frames[:num_frames]

# ---------- FEATURE EXTRACTION ----------
def extract_features_from_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Extract CNN features from frames using GoogLeNet
    
    Args:
        frames: List of frames (BGR format)
        
    Returns:
        Feature array of shape (num_frames, 1024)
    """
    _, googlenet, transform = load_models()
    
    features = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Transform and extract features
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = googlenet(image_tensor)
            features.append(feature.cpu().numpy().squeeze())
    
    return np.array(features)

# ---------- GRADCAM GENERATION ----------
def generate_gradcam_frames(frames: List[np.ndarray], num_samples: int = 6) -> List[np.ndarray]:
    """
    Generate GradCAM visualizations for sample frames
    
    Args:
        frames: List of frames
        num_samples: Number of frames to visualize
        
    Returns:
        List of GradCAM overlay images
    """
    _, googlenet, transform = load_models()
    
    # Select evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
    gradcam_results = []
    
    for idx in indices:
        frame = frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate GradCAM
        gradients = []
        activations = []
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        target_layer = googlenet.inception5b
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward pass
        output = googlenet(image_tensor)
        pred_class = output.argmax()
        
        # Backward pass
        googlenet.zero_grad()
        output[0, pred_class].backward()
        
        # Compute CAM
        gradient = gradients[0].cpu().data.numpy()[0]
        activation = activations[0].cpu().data.numpy()[0]
        weights = np.mean(gradient, axis=(1, 2))
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activation[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        # Create overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        original = cv2.resize(frame, (224, 224))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        gradcam_results.append(overlay)
        
        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()
    
    return gradcam_results


# ---------- MAIN INFERENCE FUNCTION ----------
def predict_video(video_path: str, generate_gradcam: bool = True) -> Dict:
    """
    Main inference function for video deepfake detection
    
    Args:
        video_path: Path to video file
        generate_gradcam: Whether to generate GradCAM visualizations (default: True)
        
    Returns:
        Dictionary containing:
            - prediction: "Real" or "Fake"
            - confidence: Confidence score (0-100)
            - probabilities: Dict with 'real' and 'fake' probabilities
            - gradcam_frames: List of GradCAM overlay images (if generate_gradcam=True)
            - frame_count: Number of frames analyzed
    """
    try:
        # Load models
        lstm_model, _, _ = load_models()
        
        # Step 1: Extract frames
        print(f"Extracting frames from: {video_path}")
        frames = extract_frames_from_video(video_path, num_frames=FRAME_COUNT)
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted from the video")
        
        # Step 2: Extract CNN features
        print(f"Extracting features from {len(frames)} frames...")
        features = extract_features_from_frames(frames)
        
        # Step 3: Prepare input for LSTM
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Step 4: Run LSTM prediction
        print("Running LSTM prediction...")
        with torch.no_grad():
            output = lstm_model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            real_prob = probabilities[0, 0].item()
            fake_prob = probabilities[0, 1].item()
        
        # Step 5: Determine prediction
        # Using threshold of 0.5 for softmax output
        if fake_prob > 0.5:
            prediction = "Fake"
            confidence = fake_prob * 100
        else:
            prediction = "Real"
            confidence = real_prob * 100
        
        # Step 6: Generate GradCAM if requested
        gradcam_frames = None
        if generate_gradcam:
            print("Generating GradCAM visualizations...")
            gradcam_frames = generate_gradcam_frames(frames, num_samples=6)
        
        # Prepare result
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "real": round(real_prob * 100, 2),
                "fake": round(fake_prob * 100, 2)
            },
            "gradcam_frames": gradcam_frames,
            "frame_count": len(frames)
        }
        
        print(f"Prediction: {prediction} (Confidence: {confidence:.2f}%)")
        return result
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


# ---------- TESTING ----------
if __name__ == "__main__":
    """Test the inference pipeline"""
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default test video
        video_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo_media", "videos", "fake", "fake_1.mp4"
        )
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("\nUsage: python inference.py <path_to_video>")
        sys.exit(1)
    
    print("=" * 60)
    print("VIDEO DEEPFAKE DETECTION - INFERENCE TEST")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Device: {device}")
    print(f"Model: {VIDEO_MODEL_PATH}")
    print("-" * 60)
    
    # Run prediction
    result = predict_video(video_path, generate_gradcam=False)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Probabilities:")
    print(f"  - Real: {result['probabilities']['real']}%")
    print(f"  - Fake: {result['probabilities']['fake']}%")
    print(f"Frames analyzed: {result['frame_count']}")
    print("=" * 60)
