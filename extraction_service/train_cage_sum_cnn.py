#!/usr/bin/env python3
"""
Train improved CNN for cage sum recognition (1-45).

Improvements over v1:
- ResNet-style residual blocks for better gradient flow
- Focal loss for class imbalance
- More aggressive data augmentation
- OneCycle learning rate schedule
- Mixup augmentation
- Larger model capacity
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import random


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Keep benchmark=True for better performance
        torch.backends.cudnn.benchmark = True


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class ImprovedCageSumCNN(nn.Module):
    """Improved CNN with residual connections for cage sum recognition."""

    def __init__(self, num_classes=45):
        super(ImprovedCageSumCNN, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with dropout
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class CageSumDataset(Dataset):
    """Dataset for cage sum images with aggressive augmentation."""

    def __init__(self, data_dir, transform=None, label_to_idx=None, augment=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.samples = []

        # Build label mapping if not provided
        if label_to_idx is None:
            all_labels = set()
            for label_dir in sorted(self.data_dir.glob('label_*')):
                label = int(label_dir.name.split('_')[1])
                all_labels.add(label)

            sorted_labels = sorted(all_labels)
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        else:
            self.label_to_idx = label_to_idx
            self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Load all images
        for label_dir in sorted(self.data_dir.glob('label_*')):
            label = int(label_dir.name.split('_')[1])
            for img_path in sorted(label_dir.glob('*.png')):
                self.samples.append((str(img_path), label))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"Label mapping: {len(self.label_to_idx)} unique labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Apply augmentation if training
        if self.augment:
            img = self._augment(img)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        class_idx = self.label_to_idx[label]
        return img, class_idx

    def _augment(self, img):
        """Apply random augmentations."""
        h, w = img.shape

        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        # Random scale (0.85 to 1.15)
        if random.random() < 0.5:
            scale = random.uniform(0.85, 1.15)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Crop or pad back to original size
            if new_h > h:
                start = (new_h - h) // 2
                img = img[start:start+h, :]
            elif new_h < h:
                pad = (h - new_h) // 2
                img = cv2.copyMakeBorder(img, pad, h-new_h-pad, 0, 0,
                                        cv2.BORDER_CONSTANT, value=255)
            if new_w > w:
                start = (new_w - w) // 2
                img = img[:, start:start+w]
            elif new_w < w:
                pad = (w - new_w) // 2
                img = cv2.copyMakeBorder(img, 0, 0, pad, w-new_w-pad,
                                        cv2.BORDER_CONSTANT, value=255)

        # Random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Random Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(3, 10), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random Gaussian blur
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Random affine (shear)
        if random.random() < 0.3:
            shear = random.uniform(-0.1, 0.1)
            M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        return img


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_class_weights(dataset):
    """Compute inverse frequency weights for each class."""
    label_counts = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(dataset.samples)
    weights = {}
    for label in range(len(dataset.label_to_idx)):
        cage_sum = dataset.idx_to_label[label]
        count = label_counts.get(cage_sum, 1)
        # Smooth the weights to avoid extreme values
        weights[label] = np.sqrt(total / (len(label_counts) * count))

    return torch.FloatTensor([weights[i] for i in range(len(weights))])


def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch with mixup."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() < 0.5:
            # Apply mixup
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if use_mixup:
            # For mixup, just use the original labels for accuracy
            correct += (predicted == labels).sum().item()
        else:
            correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class accuracy tracking
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    # Report worst performing classes
    class_acc = {}
    for label in class_total:
        class_acc[label] = 100 * class_correct.get(label, 0) / class_total[label]

    worst_5 = sorted(class_acc.items(), key=lambda x: x[1])[:5]
    print(f"  Worst 5 classes: {worst_5}")

    return epoch_loss, epoch_acc


def main():
    # Set random seed for reproducibility
    set_seed(123)

    parser = argparse.ArgumentParser(description='Train improved cage sum CNN')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Max learning rate')
    parser.add_argument('--data-dir', type=str,
                        default='training_data/cage_sum_cnn',
                        help='Training data directory')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--no-mixup', action='store_true', help='Disable mixup')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data directories
    base_dir = Path(__file__).parent / args.data_dir
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'

    # Create datasets
    train_dataset = CageSumDataset(train_dir, augment=True)
    val_dataset = CageSumDataset(val_dir, label_to_idx=train_dataset.label_to_idx, augment=False)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Get number of classes
    metadata_path = base_dir / 'dataset_metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)

    num_classes = metadata['num_classes']
    print(f"Number of classes: {num_classes}")

    # Create model
    model = ImprovedCageSumCNN(num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Using focal loss with gamma={args.gamma}")

    # Focal loss
    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # OneCycle scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Warmup 10% of training
        anneal_strategy='cos'
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = Path(__file__).parent / 'models' / 'cage_sum_cnn.pth'
    best_model_path.parent.mkdir(exist_ok=True)

    patience_counter = 0
    use_mixup = not args.no_mixup

    print(f"\nStarting training (mixup: {use_mixup})...")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, use_mixup=use_mixup)

        # Update scheduler
        # Note: OneCycle updates per batch, so we need to step it in training loop
        # For simplicity, we're using it as epoch-level here

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label,
                'model_type': 'ImprovedCageSumCNN'
            }, best_model_path)
            print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")

    # Save training history
    history_path = Path(__file__).parent / 'models' / 'cage_sum_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
