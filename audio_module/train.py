"""Training script for audio deepfake detector."""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AUDIO_MODEL_PATH
from audio_module.dataset import AudioDeepfakeDataset
from audio_module.model import AudioDeepfakeCNN


BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 3  # Stop if val_acc doesn't improve for 3 epochs


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train" if train_mode else "Val", leave=False)
    for x, y in pbar:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        batch_size = y.shape[0]
        total += batch_size
        total_loss += loss.item() * batch_size
        correct += (preds == y).sum().item()

        pbar.set_postfix({
            "loss": f"{(total_loss / total):.4f}",
            "acc": f"{(100.0 * correct / total):.2f}%",
        })

    return total_loss / total, 100.0 * correct / total


def main() -> None:
    print("=" * 70)
    print("AUDIO DEEPFAKE DETECTION - TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("-" * 70)

    train_ds = AudioDeepfakeDataset(split="training")
    val_ds = AudioDeepfakeDataset(split="validation")

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # Calculate class weights to handle imbalance
    train_labels = [label for _, label in train_ds.samples]
    num_real = train_labels.count(0)
    num_fake = train_labels.count(1)
    pos_weight = num_real / (num_fake + 1e-8)  # Weight for fake class (label 1)
    print(f"Class distribution - Real: {num_real}, Fake: {num_fake}")
    print(f"Positive weight (fake) for loss: {pos_weight:.4f}")
    print("-" * 70)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AudioDeepfakeCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    os.makedirs(os.path.dirname(AUDIO_MODEL_PATH), exist_ok=True)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 70)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Learning rate scheduling based on validation accuracy
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), AUDIO_MODEL_PATH)
            print(f"Saved best model -> {AUDIO_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    elapsed_min = (time.time() - start) / 60.0
    print("\n" + "=" * 70)
    print(f"Training complete in {elapsed_min:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model path: {AUDIO_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
