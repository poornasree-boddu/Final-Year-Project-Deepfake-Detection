"""Audio preprocessing utilities for deepfake detection."""

from __future__ import annotations

import numpy as np
import librosa


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    duration: float = 2.0,
) -> np.ndarray:
    """Load mono audio, then trim/pad to fixed duration."""
    signal, _ = librosa.load(audio_path, sr=sample_rate, mono=True, duration=duration)

    target_len = int(sample_rate * duration)
    if signal.shape[0] < target_len:
        signal = np.pad(signal, (0, target_len - signal.shape[0]), mode="constant")
    else:
        signal = signal[:target_len]

    return signal.astype(np.float32)


def audio_to_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    duration: float = 2.0,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Convert an audio file to a normalized log-mel spectrogram."""
    signal = load_audio(audio_path, sample_rate=sample_rate, duration=duration)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_min = float(mel_db.min())
    mel_max = float(mel_db.max())
    if mel_max - mel_min < 1e-8:
        mel_norm = np.zeros_like(mel_db, dtype=np.float32)
    else:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)

    return mel_norm.astype(np.float32)
