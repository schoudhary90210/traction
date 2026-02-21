#!/usr/bin/env python3
"""
================================================================================
 Agri-Scout — Offline Tractor-Mounted Crop Disease Detector
 Training Script for Qualcomm Edge AI Hackathon
================================================================================
 Target Hardware : Snapdragon X Elite NPU (via Qualcomm AI Hub)
 Model           : MobileNetV2 (transfer-learned, 4-class classifier)
 Export Format   : ONNX (opset 14) — ready for QNN compilation
 Dataset         : PlantVillage corn subset (4 classes, pre-split)
================================================================================
"""

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --- FIX FOR MACOS SSL ERROR ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ==============================================================================
#  1. CONFIGURATION
# ==============================================================================

CONFIG = {
# Paths
    "train_dir": "./dataset/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    "valid_dir": "./dataset/archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    "onnx_export_path": "./models/agri_scout_mobilenetv2.onnx",

    # Class labels (must match subfolder names exactly)
    "classes": [
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
    ],
    "num_classes": 4,

    # Training hyper-parameters
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_workers": 4,

    # Input spec (MobileNetV2 default)
    "input_size": 224,

    # ImageNet channel-wise statistics for normalization
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],

    # ONNX export settings
    "onnx_opset": 14,
}


# ==============================================================================
#  2. DATA AUGMENTATION & DATALOADERS
# ==============================================================================

def build_transforms(cfg: dict) -> dict:
    """Return a dict of train / valid torchvision transform pipelines."""

    train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(cfg["input_size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # --- Tractor dashcam–specific augmentations ---
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0)),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.25,
            hue=0.05,
        ),

        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["imagenet_mean"], std=cfg["imagenet_std"]),
    ])

    valid_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["imagenet_mean"], std=cfg["imagenet_std"]),
    ])

    return {"train": train_tfm, "valid": valid_tfm}


def build_dataloaders(cfg: dict) -> tuple:
    """Construct ImageFolder datasets and DataLoaders for train & valid splits."""

    tfms = build_transforms(cfg)

    train_dataset = datasets.ImageFolder(root=cfg["train_dir"], transform=tfms["train"])
    valid_dataset = datasets.ImageFolder(root=cfg["valid_dir"], transform=tfms["valid"])

    discovered = sorted(train_dataset.classes)
    expected = sorted(cfg["classes"])
    assert discovered == expected, (
        f"Class mismatch!\n  Expected : {expected}\n  Found    : {discovered}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    print(f"[DATA]  Training samples   : {len(train_dataset)}")
    print(f"[DATA]  Validation samples : {len(valid_dataset)}")
    print(f"[DATA]  Classes            : {train_dataset.classes}")
    print(f"[DATA]  Class-to-idx       : {train_dataset.class_to_idx}\n")

    return train_loader, valid_loader


# ==============================================================================
#  3. MODEL CONSTRUCTION
# ==============================================================================

def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Return a MobileNetV2 model with a frozen backbone and a fresh 4-class head."""

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] MobileNetV2 loaded (ImageNet pretrained)")
    print(f"[MODEL] Total params       : {total_params:,}")
    print(f"[MODEL] Trainable params   : {trainable_params:,}  "
          f"({trainable_params / total_params * 100:.1f}%)\n")

    return model


# ==============================================================================
#  4. TRAINING & VALIDATION LOOP
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """Run one full training epoch. Returns (avg_loss, accuracy)."""

    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, dim=1)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Evaluate the model on the validation set. Returns (avg_loss, accuracy)."""

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> nn.Module:
    """Full training driver. Returns the best model (by validation accuracy)."""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
    )

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    header = (
        f"{'Epoch':>7} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>10} | {'Val Acc':>9} | {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        elapsed = time.time() - t0

        marker = " *" if valid_acc > best_acc else ""
        print(
            f"  {epoch}/{cfg['epochs']}   | "
            f"  {train_loss:.4f}   | "
            f" {train_acc * 100:6.2f}%  | "
            f"  {valid_loss:.4f}   | "
            f" {valid_acc * 100:6.2f}%  | "
            f"{elapsed:5.1f}s{marker}"
        )

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\n[TRAIN] Best validation accuracy: {best_acc * 100:.2f}%\n")

    model.load_state_dict(best_weights)
    return model


# ==============================================================================
#  5. ONNX EXPORT
# ==============================================================================

def export_to_onnx(model: nn.Module, cfg: dict, device: torch.device) -> None:
    export_path = Path(cfg["onnx_export_path"])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. SAVE THE PYTORCH WEIGHTS FIRST (So you never have to retrain!)
    pth_path = export_path.with_suffix('.pth')
    torch.save(model.state_dict(), pth_path)
    print(f"\n[SAVE] PyTorch weights safely backed up to -> {pth_path}")

    # 2. ATTEMPT ONNX EXPORT (Using Opset 18 to match your Torch version)
    model.eval()
    model.to("cpu") 
    dummy_input = torch.randn(1, 3, cfg["input_size"], cfg["input_size"])

    print(f"[ONNX] Exporting to {export_path}...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=18,  # Bumped to 18 as demanded by your Torch version
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        file_size_mb = export_path.stat().st_size / (1024 * 1024)
        print(f"[ONNX]  File size      : {file_size_mb:.2f} MB")
        
        if file_size_mb < 1.0:
            print("❌ ERROR: Export file is too small.")
        else:
            print(f"✅ SUCCESS: Model is healthy and ready for AI Hub.")
            
    except Exception as e:
        print(f"❌ ONNX Export Failed: {e}")
        print("But don't worry, your .pth weights are safely saved!")

# ==============================================================================
#  6. MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    print("=" * 68)
    print("  Agri-Scout  |  MobileNetV2 Training Pipeline")
    print("  Target: Snapdragon X Elite NPU  •  Format: ONNX (opset 14)")
    print("=" * 68 + "\n")

    # Select compute device (Apple Silicon MPS prioritized)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}\n")

    train_loader, valid_loader = build_dataloaders(CONFIG)
    model = build_model(num_classes=CONFIG["num_classes"], device=device)
    model = train_model(model, train_loader, valid_loader, CONFIG, device)
    export_to_onnx(model, CONFIG, device)


if __name__ == "__main__":
    main()