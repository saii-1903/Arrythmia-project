#!/usr/bin/env python3
"""
evaluate_detailed.py

Comprehensive evaluation script that shows:
- Overall accuracy
- Per-class sensitivity (recall)
- Per-class specificity
- Per-class precision
- Per-class F1-score
- Confusion matrix
- Support (number of samples per class)
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import ECGRawDatasetSQL, CLASS_NAMES
from models import CNNTransformerClassifier


def collate_fn(batch):
    """Collate function for DataLoader"""
    xs = torch.stack([torch.from_numpy(b["signal"]).unsqueeze(0) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys


def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate detailed per-class metrics including sensitivity and specificity
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    metrics = {}
    
    for i, cls in enumerate(class_names):
        # Calculate TP, TN, FP, FN for this class (one-vs-all)
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # Support (number of true samples)
        support = tp + fn
        
        metrics[cls] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'support': support,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    return metrics, cm


def print_evaluation_report(y_true, y_pred, class_names):
    """Print comprehensive evaluation report"""
    
    metrics, cm = calculate_per_class_metrics(y_true, y_pred, class_names)
    
    print("\n" + "="*100)
    print("MODEL PERFORMANCE EVALUATION - DETAILED REPORT")
    print("="*100)
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Samples: {len(y_true)}")
    print(f"Number of Classes: {len(class_names)}")
    
    # Overall accuracy
    overall_acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # Per-class metrics table
    print("\n" + "="*100)
    print("PER-CLASS PERFORMANCE METRICS")
    print("="*100)
    print(f"{'Class Name':<40} | {'Sens':<8} | {'Spec':<8} | {'Prec':<8} | {'F1':<8} | {'Support':<8}")
    print("-"*100)
    
    for cls in class_names:
        m = metrics[cls]
        print(f"{cls:<40} | {m['sensitivity']:.4f}   | {m['specificity']:.4f}   | "
              f"{m['precision']:.4f}   | {m['f1']:.4f}   | {m['support']:<8}")
    
    print("-"*100)
    
    # Calculate macro and weighted averages
    total_support = sum(metrics[cls]['support'] for cls in class_names)
    
    macro_sens = np.mean([metrics[cls]['sensitivity'] for cls in class_names])
    macro_spec = np.mean([metrics[cls]['specificity'] for cls in class_names])
    macro_prec = np.mean([metrics[cls]['precision'] for cls in class_names])
    macro_f1 = np.mean([metrics[cls]['f1'] for cls in class_names])
    
    weighted_sens = sum(metrics[cls]['sensitivity'] * metrics[cls]['support'] for cls in class_names) / total_support
    weighted_spec = sum(metrics[cls]['specificity'] * metrics[cls]['support'] for cls in class_names) / total_support
    weighted_prec = sum(metrics[cls]['precision'] * metrics[cls]['support'] for cls in class_names) / total_support
    weighted_f1 = sum(metrics[cls]['f1'] * metrics[cls]['support'] for cls in class_names) / total_support
    
    print(f"{'Macro Average':<40} | {macro_sens:.4f}   | {macro_spec:.4f}   | "
          f"{macro_prec:.4f}   | {macro_f1:.4f}   | {total_support:<8}")
    print(f"{'Weighted Average':<40} | {weighted_sens:.4f}   | {weighted_spec:.4f}   | "
          f"{weighted_prec:.4f}   | {weighted_f1:.4f}   | {total_support:<8}")
    
    print("="*100)
    
    # Detailed breakdown for each class
    print("\n" + "="*100)
    print("DETAILED PER-CLASS BREAKDOWN")
    print("="*100)
    
    for cls in class_names:
        m = metrics[cls]
        if m['support'] == 0:
            continue
            
        print(f"\n{cls}:")
        print(f"  Support: {m['support']} samples")
        print(f"  True Positives (TP): {m['tp']}")
        print(f"  True Negatives (TN): {m['tn']}")
        print(f"  False Positives (FP): {m['fp']}")
        print(f"  False Negatives (FN): {m['fn']}")
        print(f"  Sensitivity (Recall): {m['sensitivity']:.4f} - Ability to detect this arrhythmia")
        print(f"  Specificity: {m['specificity']:.4f} - Ability to correctly identify non-cases")
        print(f"  Precision: {m['precision']:.4f} - Accuracy of positive predictions")
        print(f"  F1-Score: {m['f1']:.4f} - Harmonic mean of precision and recall")
    
    print("\n" + "="*100)
    
    # Confusion matrix
    print("\nCONFUSION MATRIX")
    print("="*100)
    print("Rows = True Labels, Columns = Predicted Labels\n")
    
    # Print header
    print(f"{'True \\ Pred':<20}", end="")
    for i, cls in enumerate(class_names):
        print(f"{i:>6}", end="")
    print()
    
    # Print matrix
    for i, cls in enumerate(class_names):
        print(f"{i:<3} {cls:<16}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>6}", end="")
        print()
    
    print("\n" + "="*100)


def evaluate():
    """Main evaluation function"""
    print("="*100)
    print("STARTING MODEL EVALUATION")
    print("="*100)
    
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    print("\nLoading dataset from SQL database...")
    dataset = ECGRawDatasetSQL()
    print(f"Dataset size: {len(dataset)} segments")
    
    if len(dataset) == 0:
        print("❌ No data found in SQL database.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 3. Load Model
    ckpt_path = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"\nLoading model from {ckpt_path}...")
    model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(ckpt_path, map_location=device)
    
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
        if "epoch" in state:
            print(f"Model trained for {state['epoch']} epochs")
        if "balanced_acc" in state:
            print(f"Best balanced accuracy during training: {state['balanced_acc']:.4f}")
    else:
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()

    # 4. Run Inference
    y_true = []
    y_pred = []
    
    print("\nRunning inference on all samples...")
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", ncols=80):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # 5. Print comprehensive report
    print_evaluation_report(y_true, y_pred, CLASS_NAMES)
    
    # 6. Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = BASE_DIR / "models_training" / "outputs" / "logs" / f"evaluation_report_{timestamp}.txt"
    
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(report_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print_evaluation_report(y_true, y_pred, CLASS_NAMES)
    sys.stdout = original_stdout
    
    print(f"\n✅ Evaluation report saved to: {report_file}")
    print("\n" + "="*100)
    print("EVALUATION COMPLETED")
    print("="*100)


if __name__ == "__main__":
    evaluate()
