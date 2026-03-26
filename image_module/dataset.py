"""
Image Deepfake Dataset Loader

PyTorch dataset class for loading deepfake image detection data
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional
import sys

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_DATASET_PATH


class DeepfakeImageDataset(Dataset):
    """
    Dataset class for deepfake image detection
    
    Args:
        root_dir: Root directory of the dataset
        split: 'Train', 'Validation', or 'Test'
        transform: Optional torchvision transforms
    """
    
    def __init__(
        self, 
        root_dir: str = IMAGE_DATASET_PATH,
        split: str = 'Train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Paths
        self.split_dir = os.path.join(root_dir, split)
        self.real_dir = os.path.join(self.split_dir, 'Real')
        self.fake_dir = os.path.join(self.split_dir, 'Fake')
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        # Load real images (label = 0)
        if os.path.exists(self.real_dir):
            for img_name in os.listdir(self.real_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(self.real_dir, img_name))
                    self.labels.append(0)
        
        # Load fake images (label = 1)
        if os.path.exists(self.fake_dir):
            for img_name in os.listdir(self.fake_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(self.fake_dir, img_name))
                    self.labels.append(1)
        
        print(f"Loaded {len(self)} images from {split} set")
        print(f"  - Real: {self.labels.count(0)}")
        print(f"  - Fake: {self.labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def get_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or evaluation
    
    Args:
        augment: Whether to apply data augmentation (for training)
        
    Returns:
        Composed transforms
    """
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    """Test the dataset loader"""
    print("=" * 60)
    print("Testing Image Dataset Loader")
    print("=" * 60)
    
    # Test all splits
    for split in ['Train', 'Validation', 'Test']:
        print(f"\n{split} Set:")
        dataset = DeepfakeImageDataset(split=split, transform=get_transforms(augment=False))
        print(f"Total images: {len(dataset)}")
        
        # Test loading one sample
        if len(dataset) > 0:
            img, label = dataset[0]
            print(f"Sample shape: {img.shape}")
            print(f"Sample label: {'Fake' if label == 1 else 'Real'}")
    
    print("\n" + "=" * 60)
    print("Dataset loading successful!")
