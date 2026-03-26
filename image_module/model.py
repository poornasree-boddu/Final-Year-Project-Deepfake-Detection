"""
Image Deepfake Detection Model

EfficientNet-B0 based binary classifier for deepfake image detection
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ImageDeepfakeDetector(nn.Module):
    """
    Deepfake image detector using EfficientNet-B0 backbone
    
    Args:
        pretrained: Whether to use pretrained ImageNet weights
        dropout: Dropout rate for classifier
    """
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super(ImageDeepfakeDetector, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get number of features from the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom binary classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1)  # Binary output (logit)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.backbone(x)


def create_model(device: torch.device, pretrained: bool = True) -> ImageDeepfakeDetector:
    """
    Create and initialize the model
    
    Args:
        device: Device to load model on
        pretrained: Whether to use pretrained weights
        
    Returns:
        Model instance
    """
    model = ImageDeepfakeDetector(pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == "__main__":
    """Test the model"""
    print("=" * 60)
    print("Testing Image Deepfake Detection Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(device, pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output.squeeze().cpu().numpy()}")
    
    # Test with sigmoid
    probs = torch.sigmoid(output)
    print(f"Probabilities: {probs.squeeze().cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("Model test successful!")
