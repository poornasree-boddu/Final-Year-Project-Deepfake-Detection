"""Inference pipeline for audio deepfake detection."""

from __future__ import annotations

import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_MODEL_PATH
from audio_module.model import AudioDeepfakeCNN
from audio_module.preprocess import audio_to_mel_spectrogram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def load_model() -> AudioDeepfakeCNN:
    """Load trained audio model as a singleton."""
    global _model
    if _model is None:
        model = AudioDeepfakeCNN()
        model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        _model = model
    return _model


def predict_audio(audio_path: str) -> Dict:
    """Predict whether an audio clip is real or fake."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model()

    mel = audio_to_mel_spectrogram(audio_path)
    x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        fake_prob = torch.sigmoid(logits).item()

    real_prob = 1.0 - fake_prob
    if fake_prob > 0.5:
        prediction = "Fake"
        confidence = fake_prob
    else:
        prediction = "Real"
        confidence = real_prob

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100.0, 2),
        "probabilities": {
            "real": round(real_prob * 100.0, 2),
            "fake": round(fake_prob * 100.0, 2),
        },
    }
