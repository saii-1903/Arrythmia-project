#!/usr/bin/env python3
"""
train.py

SQL-based training script for CNN+Transformer ECG arrhythmia classifier.

- Reads data from PostgreSQL via ECGRawDatasetSQL
- Handles class imbalance with:
    - WeightedRandomSampler for training batches
    - Class-weighted CrossEntropyLoss
- Saves best checkpoint to: outputs/checkpoints/best_model.pth
"""

import os
import json
import psycopg2
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

# ------------------------------
# Import class definitions
# ------------------------------
from data_loader import CLASS_NAMES, normalize_label, CLASS_INDEX
from models import CNNTransformerClassifier


# ---------------------------------------------------------------------
# SQL Dataset Class (CRITICAL FIX)
# ---------------------------------------------------------------------
class ECGRawDatasetSQL(torch.utils.data.Dataset):
    """
    Loads ECG segments directly from PostgreSQL.
    Returns:
        - signal: np.float32 (2500,)
        - label: int (index in CLASS_NAMES)
        - meta : features_json or {}
    """

    def __init__(self, sql_limit=None):
        self.conn = psycopg2.connect(
            host="localhost",
            database="ecg_analysis",
            user="ecg_user",
            password="sais"
        )
        self.conn.autocommit = True

        with self.conn.cursor() as cur:
            query = """
                SELECT segment_id, raw_signal, arrhythmia_label, features_json
                FROM ecg_features_annotatable
                WHERE raw_signal IS NOT NULL
                  AND arrhythmia_label IS NOT NULL
            """
            if sql_limit:
                query += f" LIMIT {int(sql_limit)}"

            cur.execute(query)
            rows = cur.fetchall()

        self.samples = []
        for seg_id, raw_sig, label, feats in rows:
            if raw_sig is None:
                continue

            label_clean = normalize_label(label)

            # Only keep our SQL classes
            if label_clean not in CLASS_INDEX:
                continue

            # Load and Fix Shape
            sig = np.asarray(raw_sig, dtype=np.float32)
            
            # Resample to common 2500 length (10s @ 250Hz) standard
            TARGET_LEN = 2500
            current_len = len(sig)
            
            if current_len != TARGET_LEN and current_len > 0:
                 # Linear interpolation to resize
                 sig = np.interp(
                     np.linspace(0, current_len, TARGET_LEN),
                     np.arange(current_len),
                     sig
                 ).astype(np.float32)
            elif current_len == 0:
                 sig = np.zeros(TARGET_LEN, dtype=np.float32)

            self.samples.append({
                "segment_id": seg_id,
                "signal": sig,
                "label": CLASS_INDEX[label_clean],
                "meta": feats or {}
            })

        print(f"[SQL DATASET] Loaded {len(self.samples)} usable rows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


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
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

        y_true += y.cpu().numpy().tolist()
        y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return {"loss": total_loss / len(y_true), "accuracy": acc}


# ---------------------------------------------------------------------
# Eval one epoch
# ---------------------------------------------------------------------
def eval_epoch(model, criterion, loader, device):
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
    return {"loss": total_loss / len(y_true), "accuracy": acc}


# ---------------------------------------------------------------------
# MAIN TRAIN LOOP
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset (SQL)
    dataset = ECGRawDatasetSQL(sql_limit=None)
    n_samples = len(dataset)

    if n_samples < 100:
        print("⛔ Not enough samples for training")
        return

    # Class distribution
    labels_all = [dataset[i]["label"] for i in range(n_samples)]
    counts = Counter(labels_all)

    print("\nClass counts:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx:02d} {name:30s} -> {counts.get(idx, 0)}")

    num_classes = len(CLASS_NAMES)

    # Class weights
    counts_arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts_arr[counts_arr == 0] = 1.0  # prevent div/0
    class_weights = (counts_arr.sum() / (num_classes * counts_arr))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights:", class_weights.cpu().numpy())

    # Weighted sampler
    sample_weights = [1.0 / counts_arr[lbl] for lbl in labels_all]

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
        num_samples=len(train_indices),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Loop
    best_loss = float("inf")
    ckpt_path = CHECKPOINTS / "best_model.pth"

    for ep in range(1, 16):
        print(f"\nEpoch {ep}/15")

        tr = train_epoch(model, optimizer, criterion, train_loader, device)
        va = eval_epoch(model, criterion, val_loader, device)

        print(f"Train: loss={tr['loss']:.4f}, acc={tr['accuracy']:.4f}")
        print(f"Val  : loss={va['loss']:.4f}, acc={va['accuracy']:.4f}")

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": CLASS_NAMES,
                },
                ckpt_path,
            )
            print("✅ Saved new best model")

    print("\nTraining finished.\nBest model:", ckpt_path)


if __name__ == "__main__":
    main()
