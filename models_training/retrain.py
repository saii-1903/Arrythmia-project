#!/usr/bin/env python3
"""
retrain.py

UPGRADED PRODUCTION RETRAINING SCRIPT
Incorporates:
1. Focal Loss (handles extreme imbalance better than weighted CE)
2. Oversampling minority classes via WeightedRandomSampler
3. Data augmentation for training set
4. Patient-level splitting (prevents data leakage)
5. Robust database column detection
6. Automatic file logging (TeeLogger)
"""

import os
import sys
import json
import psycopg2
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import CLASS_NAMES, normalize_label, CLASS_INDEX
from models import CNNTransformerClassifier


# ---------------------------------------------------------------------
# Logger Class for Automatic File Logging
# ---------------------------------------------------------------------
class TeeLogger:
    """Redirects stdout to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


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
    """Dataset that loads all data from SQL into RAM once, supports augmentation."""
    
    def __init__(self, sql_limit=None, augment=False):
        self.augment = augment
        self.conn_params = {
            "host": "localhost",
            "database": "ecg_analysis",
            "user": "ecg_user",
            "password": "sais"
        }
        
        print("Connecting to DB (Pre-loading optimized)...")
        self.samples = []  # List of (seg_id, label_idx, patient_id, admission_id) tuples
        self.signal_cache = {}  # Map seg_id -> numpy array
        self.has_patient_id = False  # Track if patient_id column exists
        
        with psycopg2.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Robust check for patient_id and admission_id columns
                cur.execute("SELECT * FROM ecg_features_annotatable LIMIT 0")
                available_cols = [desc[0].lower() for desc in cur.description]
                self.has_patient_id = 'patient_id' in available_cols
                has_admission_id = 'admission_id' in available_cols
                
                # Build query based on available columns
                if self.has_patient_id and has_admission_id:
                    query = """
                        SELECT segment_id, arrhythmia_label, raw_signal, patient_id, admission_id
                        FROM ecg_features_annotatable
                        WHERE raw_signal IS NOT NULL
                          AND arrhythmia_label IS NOT NULL
                          AND arrhythmia_label != 'Unlabeled'
                    """
                    print("✅ patient_id and admission_id columns found - will use for patient-level split")
                elif self.has_patient_id:
                    query = """
                        SELECT segment_id, arrhythmia_label, raw_signal, patient_id
                        FROM ecg_features_annotatable
                        WHERE raw_signal IS NOT NULL
                          AND arrhythmia_label IS NOT NULL
                          AND arrhythmia_label != 'Unlabeled'
                    """
                    print("✅ patient_id column found - will use for patient-level split")
                else:
                    query = """
                        SELECT segment_id, arrhythmia_label, raw_signal
                        FROM ecg_features_annotatable
                        WHERE raw_signal IS NOT NULL
                          AND arrhythmia_label IS NOT NULL
                          AND arrhythmia_label != 'Unlabeled'
                    """
                    print("⚠️  patient_id column NOT found - will use record-level split (DATA LEAKAGE POSSIBLE)")
                
                if sql_limit:
                    query += f" LIMIT {int(sql_limit)}"
                
                print("Executing query...")
                cur.execute(query)
                rows = cur.fetchall()
                print(f"Fetched {len(rows)} rows. Processing...")
                
                for row in rows:
                    seg_id, label, raw_sig = row[0], row[1], row[2]
                    # Extract patient_id and admission_id if available
                    patient_id = row[3] if len(row) > 3 else None
                    admission_id = row[4] if len(row) > 4 else None
                    
                    if not label:
                        continue
                    l_clean = normalize_label(label)
                    
                    if l_clean in CLASS_INDEX:
                        label_idx = CLASS_INDEX[l_clean]
                        # Store patient_id for patient-level splitting (None if not available)
                        self.samples.append((seg_id, label_idx, patient_id, admission_id))
                        
                        # Process signal immediately to memory (NO SQL IN __getitem__)
                        sig = np.array(raw_sig, dtype=np.float32)
                        
                        # Pre-resample to 2500 here to save training time
                        TARGET_LEN = 2500
                        if len(sig) != TARGET_LEN and len(sig) > 0:
                            idx_old = np.arange(len(sig))
                            idx_new = np.linspace(0, len(sig) - 1, TARGET_LEN)
                            sig = np.interp(idx_new, idx_old, sig).astype(np.float32)
                        
                        self.signal_cache[seg_id] = sig
                        
        print(f"[SQL DATASET] Indexed {len(self.samples)} segments. RAM Cache Ready.")
        print(f"[SQL DATASET] All signals pre-loaded. NO SQL connections in __getitem__.")

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
        seg_id, label_idx, patient_id, admission_id = self.samples[idx]
        sig = self.signal_cache.get(seg_id, np.zeros(2500, dtype=np.float32))
        
        if self.augment:
            sig = self._augment_signal(sig.copy())
            
        return {
            "signal": sig,
            "label": label_idx,
            "meta": {"id": seg_id, "patient_id": patient_id, "admission_id": admission_id}
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
# Train/Eval Routines
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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_true += y.cpu().numpy().tolist()
        y_pred += preds

    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return {"loss": total_loss / len(y_true), "accuracy": acc}


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
# MAIN RETRAIN LOOP
# ---------------------------------------------------------------------
def retrain_model(num_epochs=30, batch_size=32, lr=5e-4):
    # Initialize automatic logging
    os.makedirs(LOGS, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS / f"retrain_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger
    
    print(f"{'='*70}")
    print(f"RETRAINING SESSION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"{'='*70}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load full dataset
    full_dataset = ECGRawDatasetSQL(sql_limit=None, augment=False)
    n_samples = len(full_dataset)

    if n_samples < 50:
        print("⛔ Not enough samples for meaningful retraining")
        return

    labels_all = [lbl for (_, lbl, _, _) in full_dataset.samples]
    counts = Counter(labels_all)
    
    print("\nCLASS DISTRIBUTION")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx:02d} {name:30s} -> {counts.get(idx, 0)}")

    # 2. Split (Patient-level)
    patient_ids = [pid for (_, _, pid, _) in full_dataset.samples]
    unique_patients = set(patient_ids)
    
    if None in unique_patients or not full_dataset.has_patient_id:
        print("⚠️  Warning: Patient IDs missing. Falling back to stratified record-level split.")
        from sklearn.model_selection import train_test_split
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=0.15, stratify=labels_all, random_state=42
        )
    else:
        # Patient-level split
        from collections import defaultdict
        from sklearn.model_selection import train_test_split
        
        patient_to_indices = defaultdict(list)
        for idx, (_, _, pid, _) in enumerate(full_dataset.samples):
            patient_to_indices[pid].append(idx)
        
        unique_patient_list = list(unique_patients - {None})
        patient_labels = []
        for pid in unique_patient_list:
            p_labels = [labels_all[i] for i in patient_to_indices[pid]]
            patient_labels.append(Counter(p_labels).most_common(1)[0][0])
            
        train_patients, val_patients = train_test_split(
            unique_patient_list, test_size=0.15, stratify=patient_labels, random_state=42
        )
        
        train_idx, val_idx = [], []
        for pid in train_patients: train_idx.extend(patient_to_indices[pid])
        for pid in val_patients: val_idx.extend(patient_to_indices[pid])
        
        print(f"✅ Split: {len(train_patients)} train patients, {len(val_patients)} val patients")

    # 3. Create Task-Specific Datasets
    train_dataset = ECGRawDatasetSQL(sql_limit=None, augment=True)
    val_dataset = ECGRawDatasetSQL(sql_limit=None, augment=False)
    
    train_ds = torch.utils.data.Subset(train_dataset, train_idx)
    val_ds = torch.utils.data.Subset(val_dataset, val_idx)

    # 4. Focal Loss & Sampler
    num_classes = len(CLASS_NAMES)
    counts_arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts_arr[counts_arr == 0] = 1.0
    
    class_weights = np.sqrt(counts_arr.sum() / (num_classes * counts_arr))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Weighted sampler for train set
    train_labels_only = [labels_all[i] for i in train_idx]
    train_sample_weights = [(1.0 / counts_arr[lbl])**1.5 for lbl in train_labels_only]
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_idx)*2, replacement=True)

    # 5. Mistake-Aware Difficulty Calculation
    model = CNNTransformerClassifier(num_classes=num_classes).to(device)
    ckpt_path = CHECKPOINTS / "best_model.pth"
    
    sample_difficulty = np.ones(len(train_idx), dtype=np.float32)
    
    if ckpt_path.exists():
        print(f"🔄 Loading current model for 'Mistake Discovery' from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"] if "model_state" in state else state)
        
        # Audit the training set to find what the model gets wrong
        audit_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        model.eval()
        difficulties = []
        with torch.no_grad():
            for x, y, _ in tqdm(audit_loader, desc="🔍 Auditing Mistakes"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                # Loss per sample - higher loss means deeper mistake
                loss = F.cross_entropy(logits, y, reduction='none')
                difficulties.extend(loss.cpu().numpy().tolist())
        
        sample_difficulty = np.array(difficulties)
        # Normalize difficulty to avoid extreme sampling
        sample_difficulty = np.clip(sample_difficulty, 0.1, 5.0) 
        print(f"✅ Mistake discovery complete. Max difficulty: {sample_difficulty.max():.2f}")
    else:
        print("💡 No existing checkpoint found. Standard training from scratch.")

    # 6. Combined Sampler (Balance + Mistakes)
    # Combine static class weights with dynamic difficulty scores
    train_labels_only = [labels_all[i] for i in train_idx]
    
    # Final weight = (Class Frequency Weight) * (Previous Error Difficulty)
    combined_weights = []
    for i, lbl in enumerate(train_labels_only):
        class_w = (1.0 / counts_arr[lbl])**1.5
        mistake_w = sample_difficulty[i]
        combined_weights.append(class_w * mistake_w)
        
    sampler = WeightedRandomSampler(combined_weights, num_samples=len(train_idx)*2, replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 7. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # 8. Training Loop
    best_balanced_acc = 0.0
    
    for ep in range(1, num_epochs + 1):
        tr = train_epoch(model, optimizer, criterion, train_loader, device)
        va = eval_epoch(model, criterion, val_loader, device, num_classes)
        
        balanced_acc = np.mean(list(va['per_class_acc'].values()))
        print(f"Epoch {ep:02d} | Train Loss: {tr['loss']:.4f} | Val Loss: {va['loss']:.4f} | Bal Acc: {balanced_acc:.4f}")
        
        scheduler.step(va["loss"])

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "balanced_acc": balanced_acc,
                "class_names": CLASS_NAMES
            }, ckpt_path)
            print(f"  ⭐ Saved new best model (Bal Acc: {balanced_acc:.4f})")

    print(f"\n✅ Retraining finished. Best Balanced Accuracy: {best_balanced_acc:.4f}")
    
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    retrain_model(num_epochs=args.epochs, batch_size=args.batch, lr=args.lr)
