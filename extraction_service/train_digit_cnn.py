#!/usr/bin/env python3
"""
Training script for the Sudoku Digit CNN classifier.

Usage:
    python train_digit_cnn.py [--epochs N] [--batch-size N] [--lr LR]

Requirements:
    - Run extract_training_cells.py first to generate training data
    - PyTorch with CUDA support (optional but recommended)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from digit_classifier import SudokuDigitCNN


class SudokuCellDataset(Dataset):
    """Dataset for loading sudoku cell images with digit labels."""

    def __init__(self, data_dir, split='train', transform=None):
        """Initialize dataset.

        Args:
            data_dir: Path to training_data directory
            split: 'train' or 'val'
            transform: Optional torchvision transforms
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load file list
        split_file = os.path.join(data_dir, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: load all files directly
            file_list = []
            cells_dir = os.path.join(data_dir, 'cells')
            for digit in range(10):
                digit_dir = os.path.join(cells_dir, str(digit))
                if os.path.exists(digit_dir):
                    for f in os.listdir(digit_dir):
                        if f.endswith('.png'):
                            file_list.append(f"{digit}/{f}")

        # Build samples list
        cells_dir = os.path.join(data_dir, 'cells')
        for rel_path in file_list:
            digit = int(rel_path.split('/')[0])
            full_path = os.path.join(cells_dir, rel_path)
            if os.path.exists(full_path):
                self.samples.append((full_path, digit))

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return dummy if image fails to load
            img = np.zeros((64, 64), dtype=np.uint8)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor [1, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label


class TrainingTransforms:
    """Training-time augmentations using torchvision."""

    @staticmethod
    def get_train_transforms():
        return transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])

    @staticmethod
    def get_val_transforms():
        return None  # No augmentation for validation


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validate the model.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_model(data_dir, output_dir, epochs=100, batch_size=32, lr=0.001,
                patience=15, min_epochs=20):
    """Train the digit classifier.

    Args:
        data_dir: Path to training_data directory
        output_dir: Directory to save model and logs
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        patience: Early stopping patience (epochs without improvement)
        min_epochs: Minimum epochs before early stopping can trigger
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    train_dataset = SudokuCellDataset(
        data_dir, split='train',
        transform=TrainingTransforms.get_train_transforms()
    )
    val_dataset = SudokuCellDataset(
        data_dir, split='val',
        transform=TrainingTransforms.get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = SudokuDigitCNN(dropout_rate=0.5)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training state
    best_val_loss = float('inf')
    best_val_acc = 0
    epochs_without_improvement = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Check for improvement
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            improved = True
            epochs_without_improvement = 0

            # Save best model
            model_path = os.path.join(output_dir, 'digit_cnn.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"No improvement for {patience} consecutive epochs")
            break

    # Save final model and history
    final_model_path = os.path.join(output_dir, 'digit_cnn_final.pth')
    torch.save(model.state_dict(), final_model_path)

    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir}/digit_cnn.pth")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train Sudoku Digit CNN')
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Path to training data directory')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for model and logs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    if not data_dir.exists():
        print(f"ERROR: Training data directory not found: {data_dir}")
        print("Run extract_training_cells.py first to generate training data.")
        return 1

    train_model(
        str(data_dir),
        str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )

    return 0


if __name__ == '__main__':
    exit(main())
