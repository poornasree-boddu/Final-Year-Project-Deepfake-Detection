"""PyTorch dataset for audio deepfake detection."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_DATASET_PATH
from audio_module.preprocess import audio_to_mel_spectrogram


class AudioDeepfakeDataset(Dataset):
    """Loads audio files from split/fake and split/real directories."""

    def __init__(
        self,
        split: str,
        root_dir: str = AUDIO_DATASET_PATH,
    ) -> None:
        self.split = split
        self.root_dir = root_dir

        self.samples: List[Tuple[str, int]] = []
        split_dir = os.path.join(root_dir, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for label_name, label_value in (("real", 0), ("fake", 1)):
            class_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(".wav"):
                    self.samples.append((os.path.join(class_dir, file_name), label_value))

        if not self.samples:
            raise ValueError(f"No .wav files found in split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        mel = audio_to_mel_spectrogram(path)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y
