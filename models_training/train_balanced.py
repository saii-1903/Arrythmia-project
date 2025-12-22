#!/usr/bin/env python3
"""
train_balanced.py

IMPROVED training script with aggressive class balancing strategies:
1. Focal Loss (handles extreme imbalance better than weighted CE)
2. Oversampling minority classes
3. Data augmentation for minority classes
4. Adjusted learning rate and regularization
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

from data_loader import CLASS_NAMES, normalize_label, CLASS_INDEX
from models import CNNTransformerClassifier


# ---------------------------------------------------------------------
# FOCAL LOSS - Better for extreme imbalance
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss: focuses on hard examples and down-weights easy ones.
    Better than weighted CE for extreme imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma  # focusing parameter
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------
# SQL Dataset with Data Augmentation
# ---------------------------------------------------------------------
class ECGRawDatasetSQL(torch.utils.data.Dataset):
    def __init__(self, sql_limit=None, augment=False):
        self.augment = augment
        self.conn_params = {
            "host": "localhost",
            "database": "ecg_analysis",
            "user": "ecg_user",
            "password": "sais"
        }
        
        print("Connecting to DB (Lazy Load Mode)...")
        self.samples = []
        
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT segment_id, arrhythmia_label 
                    FROM ecg_features_annotatable
                    WHERE raw_signal IS NOT NULL
                      AND arrhythmia_label IS NOT NULL
                      AND arrhythmia_label != 'Unlabeled'
                """
                if sql_limit:
                    query += f" LIMIT {int(sql_limit)}"
                
                cur.execute(query)
                rows = cur.fetchall()
                
                for seg_id, label in rows:
                    if not label: continue
                    l_clean = normalize_label(label)
                    if l_clean in CLASS_INDEX:
                        self.samples.append((seg_id, CLASS_INDEX[l_clean]))
                        
        print(f"[SQL DATASET] Indexed {len(self.samples)} segments for training")

    def __len__(self):
        return len(self.samples)

    def _augment_signal(self, signal):
        """Simple augmentation: random scaling and noise"""
        if not self.augment or np.random.rand() > 0.5:
            return signal
            
        # Random amplitude scaling (0.8 to 1.2)
        scale = np.random.uniform(0.8, 1.2)
        signal = signal * scale
        
        # Add small Gaussian noise
        sigma = 0.02 * np.std(signal)
        if sigma > 0:
            noise = np.random.normal(0, sigma, signal.shape)
            signal = signal + noise
        
        return signal.astype(np.float32)

    def __getitem__(self, idx):
        seg_id, label_idx = self.samples[idx]
        
        # Lazy load signal
        sig = np.zeros(2500, dtype=np.float32)
        
        conn = psycopg2.connect(**self.conn_params)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT raw_signal FROM ecg_features_annotatable WHERE segment_id = %s", (seg_id,))
                row = cur.fetchone()
                if row and row[0] is not None:
                     sig = np.array(row[0], dtype=np.float32)
        except Exception as e:
            print(f"Error loading segment {seg_id}: {e}")
        finally:
            conn.close()

        # Resample logic
        TARGET_LEN = 2500
        current_len = len(sig)
        
        if current_len != TARGET_LEN and current_len > 0:
             idx_old = np.arange(current_len)
             idx_new = np.linspace(0, current_len - 1, TARGET_LEN)
             sig = np.interp(idx_new, idx_old, sig).astype(np.float32)
        elif current_len == 0:
             # Already zeros
             pass

        if self.augment:
            sig = self._augment_signal(sig)
            
        return {
            "signal": sig,
            "label": label_idx,
            "meta": {"id": seg_id}
        }


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------
OUTPUT = Path("outputs")
CHECKPOINTS = OUTPUT / "checkpoints"
LOGS = OUTPUT / "logs"

for d in (OUTPUT, CHECKPOINTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------
def collate_fn(batch):
    xs = torch.stack(
        [torch.from_numpy(b["signal"]).float().unsqueeze(0) for b in batch],
        dim=0,
    )
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]
    return xs, ys, metas


# ---------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------
def train_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y, metas in tqdm(loader, desc="train", ncols=80):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_true += y.cpu().numpy().tolist()
        y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return {"loss": total_loss / len(y_true), "accuracy": acc}


# ---------------------------------------------------------------------
# Eval one epoch with per-class accuracy
# ---------------------------------------------------------------------
def eval_epoch(model, criterion, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y, metas in tqdm(loader, desc="val  ", ncols=80):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            y_true += y.cpu().numpy().tolist()
            y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    
    # Per-class accuracy
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    per_class_acc = {}
    for i in range(num_classes):
        mask = y_true_arr == i
        if mask.sum() > 0:
            per_class_acc[i] = float((y_pred_arr[mask] == i).mean())
        else:
            per_class_acc[i] = 0.0
    
    return {
        "loss": total_loss / len(y_true), 
        "accuracy": acc,
        "per_class_acc": per_class_acc
    }


# ---------------------------------------------------------------------
# MAIN TRAIN LOOP
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset with augmentation for training
    dataset = ECGRawDatasetSQL(sql_limit=None, augment=True)
    n_samples = len(dataset)

    if n_samples < 100:
        print("⛔ Not enough samples for training")
        return

    # Class distribution
    labels_all = [dataset.samples[i]["label"] for i in range(n_samples)]
    counts = Counter(labels_all)

    print("\nClass counts:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx:02d} {name:30s} -> {counts.get(idx, 0)}")

    num_classes = len(CLASS_NAMES)

    # AGGRESSIVE class weights for focal loss
    counts_arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts_arr[counts_arr == 0] = 1.0
    
    # Use sqrt of inverse frequency for less aggressive weighting
    # (focal loss will handle the rest)
    class_weights = np.sqrt(counts_arr.sum() / (num_classes * counts_arr))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights:", class_weights.cpu().numpy())

    # VERY AGGRESSIVE weighted sampler - oversample minority classes heavily
    sample_weights = []
    for lbl in labels_all:
        # Square the inverse frequency for more aggressive oversampling
        weight = (1.0 / counts_arr[lbl]) ** 1.5
        sample_weights.append(weight)

    # Train-val split
    train_size = int(0.85 * n_samples)
    val_size = n_samples - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if hasattr(train_ds, "indices"):
        train_indices = train_ds.indices
    else:
        train_indices = range(train_size)

    sampler = WeightedRandomSampler(
        [sample_weights[i] for i in train_indices],
        num_samples=len(train_indices) * 2,  # Sample 2x to ensure minority classes seen
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    
    # Lower learning rate for better convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Use Focal Loss instead of CrossEntropy
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Loop
    best_loss = float("inf")
    best_balanced_acc = 0.0
    ckpt_path = CHECKPOINTS / "best_model.pth"

    for ep in range(1, 31):  # More epochs
        print(f"\nEpoch {ep}/30")

        tr = train_epoch(model, optimizer, criterion, train_loader, device)
        va = eval_epoch(model, criterion, val_loader, device, num_classes)

        print(f"Train: loss={tr['loss']:.4f}, acc={tr['accuracy']:.4f}")
        print(f"Val  : loss={va['loss']:.4f}, acc={va['accuracy']:.4f}")
        
        # Print per-class accuracy
        print("Per-class accuracy:")
        for idx, name in enumerate(CLASS_NAMES):
            acc = va['per_class_acc'].get(idx, 0.0)
            print(f"  {name:40s}: {acc:.3f}")
        
        # Calculate balanced accuracy (mean of per-class accuracies)
        balanced_acc = np.mean([va['per_class_acc'].get(i, 0.0) for i in range(num_classes)])
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        
        scheduler.step(va["loss"])

        # Save based on balanced accuracy, not just loss
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_loss = va["loss"]
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": CLASS_NAMES,
                    "balanced_acc": balanced_acc,
                },
                ckpt_path,
            )
            print(f"✅ Saved new best model (balanced acc: {balanced_acc:.4f})")

    print("\nTraining finished.\nBest model:", ckpt_path)
    print(f"Best balanced accuracy: {best_balanced_acc:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("training_error.log", "w") as f:
            f.write(traceback.format_exc())
        print("Training failed. See training_error.log")
        sys.exit(1)
