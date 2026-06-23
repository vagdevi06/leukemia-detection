"""
scripts/train_cnn.py
Train LeukemiaCNN on C-NMC leukemia dataset.
Uses Apple MPS GPU for fast training on Mac.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import yaml
import numpy as np

from app.models.cnn_classifier import LeukemiaCNN


def build_transforms(input_size, train=True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((input_size[0] + 32, input_size[1] + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_balanced_sampler(dataset):
    """Create a weighted sampler to balance classes."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts
    sample_weights = [weights[t] for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=0.0001)
    parser.add_argument("--backbone", default="resnet50")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    input_size  = tuple(cfg["model"]["cnn"]["input_size"])
    num_classes = cfg["model"]["cnn"]["num_classes"]

    # Datasets
    train_dir = Path("data/processed2/train")
    val_dir   = Path("data/processed2/val")

    train_ds = datasets.ImageFolder(
        root=str(train_dir),
        transform=build_transforms(input_size, train=True)
    )
    val_ds = datasets.ImageFolder(
        root=str(val_dir),
        transform=build_transforms(input_size, train=False)
    )

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Count classes
    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets)
    print(f"Class distribution: {dict(zip(train_ds.classes, class_counts))}")

    # Balanced sampler
    sampler = make_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = LeukemiaCNN(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    # Loss with class weights
    class_weights = torch.tensor(
        [1.0 / c for c in class_counts],
        dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    save_dir = Path("weights")
    save_dir.mkdir(exist_ok=True)
    best_val_acc = 0.
    patience     = 0
    start_epoch  = 1

    # Resume from checkpoint
    checkpoint_path = save_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_acc = ckpt["best_val_acc"]
        start_epoch  = ckpt["epoch"] + 1
        patience     = ckpt.get("patience", 0)
        print(f"Resumed from epoch {ckpt['epoch']} val_acc={best_val_acc:.4f}")

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"{dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(str(save_dir / "cnn_classifier.pt"))
            patience = 0
            print(f"  ✓ Best val_acc={best_val_acc:.4f} saved!")
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoint every epoch
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "patience":     patience,
        }, str(checkpoint_path))
        print(f"  Checkpoint saved at epoch {epoch}")

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to: weights/cnn_classifier.pt")


if __name__ == "__main__":
    main()