"""GradCAM generation for image deepfake inference."""

from __future__ import annotations

import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGE_MODEL_PATH
from image_module.model import ImageDeepfakeDetector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_target_layer = None
_transform = None


def _load_gradcam_model() -> Tuple[ImageDeepfakeDetector, torch.nn.Module, transforms.Compose]:
    """Load model and GradCAM target layer as singletons."""
    global _model, _target_layer, _transform

    if _model is None:
        model = ImageDeepfakeDetector(pretrained=False)
        model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        target_layer = model.backbone.features[-1]

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _model = model
        _target_layer = target_layer
        _transform = preprocess

    return _model, _target_layer, _transform


def generate_image_gradcam(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return RGB original image and RGB GradCAM overlay, both 224x224."""
    model, target_layer, transform = _load_gradcam_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(image_tensor)
        model.zero_grad(set_to_none=True)

        # Backpropagate fake-class logit (single-logit BCE setup).
        output[0, 0].backward()

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for idx, weight in enumerate(weights):
            cam += weight * acts[idx]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam_min = float(np.min(cam))
        cam_max = float(np.max(cam))
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise FileNotFoundError(f"Unable to read image for GradCAM: {image_path}")

        original_bgr = cv2.resize(original_bgr, (224, 224))
        overlay_bgr = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        return original_rgb, overlay_rgb
    finally:
        forward_handle.remove()
        backward_handle.remove()
