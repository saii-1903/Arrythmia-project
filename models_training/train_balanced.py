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
                # Check if patient_id and admission_id columns exist
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'ecg_features_annotatable'
                    AND column_name IN ('patient_id', 'admission_id')
                """)
                available_cols = [row[0] for row in cur.fetchall()]
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
        # BUG FIX: Unpack tuple correctly (seg_id, label_idx, patient_id, admission_id)
        seg_id, label_idx, patient_id, admission_id = self.samples[idx]
        
        # RAM Fetch (Super fast) - NO SQL CONNECTION HERE
        sig = self.signal_cache.get(seg_id, np.zeros(2500, dtype=np.float32))
        
        # Augment on the fly (only if self.augment=True)
        if self.augment:
            sig = self._augment_signal(sig.copy())  # copy to avoid mutating cache
            
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

    # 1. Load full dataset WITHOUT augmentation first (for proper splitting)
    print("\n" + "="*70)
    print("LOADING DATASET (no augmentation yet)")
    print("="*70)
    full_dataset = ECGRawDatasetSQL(sql_limit=None, augment=False)
    n_samples = len(full_dataset)

    if n_samples < 100:
        print("⛔ Not enough samples for training")
        return

    # BUG FIX #1: Extract labels correctly from tuples, not dicts
    # full_dataset.samples contains (seg_id, label_idx, patient_id, admission_id)
    labels_all = [lbl for (_, lbl, _, _) in full_dataset.samples]
    counts = Counter(labels_all)
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx:02d} {name:30s} -> {counts.get(idx, 0)}")

    # BUG FIX #3: Patient-level split to prevent data leakage
    print("\n" + "="*70)
    print("PATIENT-LEVEL SPLIT (preventing data leakage)")
    print("="*70)
    
    # Extract patient IDs
    patient_ids = [pid for (_, _, pid, _) in full_dataset.samples]
    unique_patients = set(patient_ids)
    
    print(f"Total samples: {n_samples}")
    print(f"Unique patients: {len(unique_patients)}")
    
    # Check if patient_id is available
    if None in unique_patients:
        print("⚠️  WARNING: patient_id not available in database!")
        print("⚠️  Falling back to record-level split (DATA LEAKAGE POSSIBLE)")
        print("⚠️  This is NOT recommended for medical ML!")
        
        # Fallback: stratified split by indices
        from sklearn.model_selection import train_test_split
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices, test_size=0.15, stratify=labels_all, random_state=42
        )
    else:
        print("✅ Patient IDs available - performing patient-level split")
        
        # Group samples by patient
        from collections import defaultdict
        patient_to_indices = defaultdict(list)
        for idx, (_, _, pid, _) in enumerate(full_dataset.samples):
            patient_to_indices[pid].append(idx)
        
        # Split patients (not samples)
        from sklearn.model_selection import train_test_split
        unique_patient_list = list(unique_patients - {None})
        
        # Get labels for each patient (use majority label)
        patient_labels = []
        for pid in unique_patient_list:
            patient_sample_labels = [labels_all[i] for i in patient_to_indices[pid]]
            majority_label = Counter(patient_sample_labels).most_common(1)[0][0]
            patient_labels.append(majority_label)
        
        train_patients, val_patients = train_test_split(
            unique_patient_list, 
            test_size=0.15, 
            stratify=patient_labels, 
            random_state=42
        )
        
        # Convert patient split to sample indices
        train_idx = []
        val_idx = []
        for pid in train_patients:
            train_idx.extend(patient_to_indices[pid])
        for pid in val_patients:
            val_idx.extend(patient_to_indices[pid])
        
        # Verify no patient overlap
        train_patients_set = set(train_patients)
        val_patients_set = set(val_patients)
        overlap = train_patients_set & val_patients_set
        
        if overlap:
            print(f"❌ CRITICAL: Patient overlap detected: {len(overlap)} patients")
            print(f"   This indicates a BUG in the splitting logic!")
        else:
            print(f"✅ No patient overlap - split is valid")
            print(f"   Train patients: {len(train_patients)}")
            print(f"   Val patients: {len(val_patients)}")
            print(f"   Train samples: {len(train_idx)}")
            print(f"   Val samples: {len(val_idx)}")
    
    # BUG FIX #2: Separate datasets for train (augmented) and val (no augmentation)
    print("\n" + "="*70)
    print("CREATING TRAIN (augmented) AND VAL (no augment) DATASETS")
    print("="*70)
    
    # Create separate dataset instances
    train_dataset = ECGRawDatasetSQL(sql_limit=None, augment=True)
    val_dataset = ECGRawDatasetSQL(sql_limit=None, augment=False)
    
    # Create Subsets with the split indices
    train_ds = torch.utils.data.Subset(train_dataset, train_idx)
    val_ds = torch.utils.data.Subset(val_dataset, val_idx)
    
    print(f"✅ Train dataset: {len(train_ds)} samples (augmentation=ON)")
    print(f"✅ Val dataset: {len(val_ds)} samples (augmentation=OFF)")

    num_classes = len(CLASS_NAMES)

    # AGGRESSIVE class weights for focal loss
    counts_arr = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts_arr[counts_arr == 0] = 1.0
    
    # Use sqrt of inverse frequency for less aggressive weighting
    # (focal loss will handle the rest)
    class_weights = np.sqrt(counts_arr.sum() / (num_classes * counts_arr))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\n" + "="*70)
    print("CLASS WEIGHTS FOR FOCAL LOSS")
    print("="*70)
    print("Class weights:", class_weights.cpu().numpy())

    # VERY AGGRESSIVE weighted sampler - oversample minority classes heavily
    print("\n" + "="*70)
    print("CREATING WEIGHTED SAMPLER")
    print("="*70)
    
    sample_weights = []
    for lbl in labels_all:
        # Square the inverse frequency for more aggressive oversampling
        weight = (1.0 / counts_arr[lbl]) ** 1.5
        sample_weights.append(weight)

    # Use train_idx to get weights only for training samples
    sampler = WeightedRandomSampler(
        [sample_weights[i] for i in train_idx],
        num_samples=len(train_idx) * 2,  # Sample 2x to ensure minority classes seen
        replacement=True
    )
    
    print(f"✅ Weighted sampler created: {len(train_idx) * 2} samples per epoch")

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
