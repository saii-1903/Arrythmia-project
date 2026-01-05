#!/usr/bin/env python3
"""
Retrain model from SQL ECG segments.
Uses ECGRawDatasetSQL + rules-compatible CLASS_NAMES.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path to allow imports from xai
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_loader import ECGRawDatasetSQL, CLASS_NAMES
from models import CNNTransformerClassifier
from xai.xai import reset_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# -----------------------------------------------------
# Checkpoints
# -----------------------------------------------------
# Ensure we save to models_training/outputs/checkpoints regardless of where script is run
CKPT_DIR = Path(__file__).parent / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "best_model.pth"


# -----------------------------------------------------
# Metrics
# -----------------------------------------------------
def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average='macro')) if len(y_true) > 0 else 0.0
    # Use the dynamic CLASS_NAMES length
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist()}


# -----------------------------------------------------
# Collate fn (required for SQL dataset)
# -----------------------------------------------------
def collate_fn(batch):
    xs = torch.stack([torch.from_numpy(b["signal"]) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys


# -----------------------------------------------------
# Epoch routines
# -----------------------------------------------------
def train_epoch(model, opt, loss_fn, loader, device):
    model.train()
    losses = []
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Train"):
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred += preds
        y_true += y.cpu().numpy().tolist()

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def eval_epoch(model, loss_fn, loader, device):
    model.eval()
    losses = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred += preds
            y_true += y.cpu().numpy().tolist()

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses))
    return metrics


# -----------------------------------------------------
# Main routine
# -----------------------------------------------------
def retrain_model(num_epochs=50):

    print("\n=== RETRAINING FROM SQL DATA ===")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    # Load dataset
    dataset = ECGRawDatasetSQL(limit=None)
    print(f"Loaded {len(dataset)} segments")

    if len(dataset) < 5:
        print("⛔ Not enough SQL data to train. Need at least 5 segments.")
        return False

    # Extract labels
    # Optimization: If dataset has 'samples' (ECGRawDatasetSQL), use that to avoid DB fetch overhead
    if hasattr(dataset, "samples"):
        labels_all = [s[1] for s in dataset.samples]
    else:
        labels_all = [dataset[i]["label"] for i in range(len(dataset))]

    # Labels are already indices (ints) from ECGRawDatasetSQL
    # No need to map CLASS_NAMES.index(lbl)
    num_classes = len(CLASS_NAMES)

    # Class weights
    counts = np.bincount(labels_all, minlength=num_classes)
    counts[counts == 0] = 1
    class_weights = (counts.sum() / (num_classes * counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Train/Val split (Stratified to ensure rare classes are in Train)
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    train_ratio = 0.85
    indices = np.arange(len(dataset))
    
    try:
        # Attempt stratified split
        train_idx, val_idx = train_test_split(
            indices, 
            train_size=train_ratio, 
            stratify=labels_all, 
            random_state=42
        )
    except ValueError:
        # Fallback if some classes have too few samples (<2)
        print("[!] Warning: Some classes have too few samples for stratified split. Falling back to random split.")
        train_idx, val_idx = train_test_split(
            indices, 
            train_size=train_ratio, 
            random_state=42
        )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    
    # Weighted sampler (re-calculate specific to TRAIN set for correctness)
    train_labels = [labels_all[i] for i in train_idx]
    train_counts = np.bincount(train_labels, minlength=num_classes)
    train_counts[train_counts == 0] = 1 # avoid div/0
    
    # Inverse frequency weights
    sample_weights = [1.0 / train_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

    # Device optimization
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"[+] CUDA Enabled! Training on {torch.cuda.get_device_name(0)}")
        use_pin_memory = True
    else:
        print("[!] CUDA not available. Training on CPU (slower).")
        use_pin_memory = False

    # Loaders (pin_memory helps transfer speed to GPU)
    train_loader = DataLoader(
        train_ds, 
        batch_size=16, 
        sampler=sampler, 
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )

    # Model + Optimizer
    print("Initializing NEW model from scratch...")
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    best_val = 1e9
    
    for ep in range(1, num_epochs + 1):
        print(f"\nEpoch {ep}/{num_epochs}")

        tr = train_epoch(model, opt, loss_fn, train_loader, device)
        va = eval_epoch(model, loss_fn, val_loader, device)

        print("Train:", tr)
        print("Val:", va)

        if va["loss"] < best_val:
            best_val = va["loss"]
            print("✓ Saving BEST model:", CKPT_PATH)
            torch.save({"model_state": model.state_dict()}, CKPT_PATH)

    reset_model()
    print("\n=== Retraining COMPLETE ===")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()
    
    retrain_model(num_epochs=args.epochs)
