"""
Retraining script that:
 - Loads original datasets
 - Adds SQL corrected JSONs
 - Retrains model
 - Saves new checkpoint as best_model.pth
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

from data_loader import ECGRawDataset, CLASS_NAMES
from models import CNNTransformerClassifier
from utils import compute_metrics

# DATA SOURCES
ORIGINAL_DATA_ROOTS = [
    Path("input_segments_mitdb"),
    Path("input_segments_afdb"),
    Path("input_segments_ptbxl"),
]

SQL_CORRECTED_DIR = Path("retraining_data")

SAVE_DIR = Path("outputs/checkpoints")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def collate_fn(batch):
    xs = torch.stack([torch.from_numpy(b["signal"]) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys


def main():

    print("Loading original datasets...")
    dataset = ECGRawDataset(
        json_dir=None,
        from_sql=False,
        data_roots=ORIGINAL_DATA_ROOTS
    )

    print("Loading corrected SQL segments...")
    sql_dataset = ECGRawDataset(
        json_dir=SQL_CORRECTED_DIR,
        from_sql=False,
        data_roots=None
    )

    print(f"Original segments: {len(dataset)}")
    print(f"Corrected SQL segments: {len(sql_dataset)}")

    # Combine datasets
    dataset.samples.extend(sql_dataset.samples)
    print(f"Total training segments: {len(dataset)}")

    labels_all = [dataset[i]["label"] for i in range(len(dataset))]
    num_classes = len(CLASS_NAMES)

    # Class distribution
    counts = np.bincount(labels_all, minlength=num_classes)
    print("Class counts:", counts)

    # Class weights
    inv_freq = (counts.sum() / (counts + 1e-6)) / num_classes
    class_weights = torch.tensor(inv_freq, dtype=torch.float32)

    # Split
    n = len(dataset)
    n_train = int(n * 0.85)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Sampler
    sample_weights = [1.0 / (counts[label] + 1e-6) for label in labels_all]
    train_weights = [sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CNNTransformerClassifier(num_classes=num_classes)
    model = model.cuda() if torch.cuda.is_available() else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = 1e9
    EPOCHS = 12

    for ep in range(1, EPOCHS+1):
        print(f"\nEpoch {ep}/{EPOCHS}")

        # Train
        model.train()
        tr_loss = []
        for x, y in tqdm(train_loader):
            x = x.cuda() if torch.cuda.is_available() else x
            y = y.cuda() if torch.cuda.is_available() else y

            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tr_loss.append(loss.item())

        # Val
        model.eval()
        val_loss = []
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.cuda() if torch.cuda.is_available() else x
                y = y.cuda() if torch.cuda.is_available() else y
                out = model(x)
                loss = crit(out, y)
                val_loss.append(loss.item())

                p = out.argmax(dim=1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(p.cpu().numpy().tolist())

        avg_vloss = np.mean(val_loss)
        metrics = compute_metrics(y_true, y_pred)

        print("Val Loss:", avg_vloss)
        print("Metrics:", metrics)

        if avg_vloss < best_loss:
            best_loss = avg_vloss
            torch.save({"model_state": model.state_dict()},
                       SAVE_DIR/"best_model.pth")
            print("Saved BEST checkpoint.")

    print("\nRetraining complete.")
    print(f"New best_model.pth saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
