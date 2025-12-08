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

from data_loader import ECGRawDatasetSQL, CLASS_NAMES
from models import CNNTransformerClassifier
from xai import reset_model
from utils import compute_metrics

# -----------------------------------------------------
# Checkpoints
# -----------------------------------------------------
CKPT_DIR = Path("outputs/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "final.pt"


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
def retrain_model():

    print("\n=== RETRAINING FROM SQL DATA ===")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset = ECGRawDatasetSQL(limit=None)
    print(f"Loaded {len(dataset)} segments")

    if len(dataset) < 100:
        print("⛔ Not enough SQL data to train.")
        return False

    # Extract labels
    labels_all = [dataset[i]["label"] for i in range(len(dataset))]

    # Convert label strings → class index
    labels_all = [CLASS_NAMES.index(lbl) for lbl in labels_all]
    num_classes = len(CLASS_NAMES)

    # Class weights
    counts = np.bincount(labels_all, minlength=num_classes)
    counts[counts == 0] = 1
    class_weights = (counts.sum() / (num_classes * counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Train/Val split
    train_ratio = 0.85
    n_train = int(train_ratio * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Weighted sampler
    sample_weights = [1.0 / counts[lbl] for lbl in labels_all]
    train_indices = train_ds.indices
    sampler_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_indices), replacement=True)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model + Optimizer
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    best_val = 1e9
    epochs = 15

    for ep in range(1, epochs + 1):
        print(f"\nEpoch {ep}/{epochs}")

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
    retrain_model()
