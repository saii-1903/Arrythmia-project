#!/usr/bin/env python3
"""
Simple evaluation script with weights_only=False for compatibility
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


def calculate_per_class_metrics(y_true, y_pred, num_classes):
    """Calculate detailed per-class metrics"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    metrics = []
    
    for i in range(num_classes):
        # Calculate TP, TN, FP, FN for this class
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        support = tp + fn
        
        metrics.append({
            'class_idx': i,
            'class_name': CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}",
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'support': support,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    return metrics, cm


def print_report(metrics, cm, total_samples):
    """Print formatted evaluation report"""
    
    print("\n" + "="*120)
    print("MODEL PERFORMANCE EVALUATION - DETAILED REPORT")
    print("="*120)
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Classes: {len(metrics)}")
    
    # Overall accuracy
    correct = sum(m['tp'] for m in metrics)
    overall_acc = correct / total_samples if total_samples > 0 else 0.0
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # Per-class metrics table
    print("\n" + "="*120)
    print("PER-CLASS PERFORMANCE METRICS")
    print("="*120)
    print(f"{'Class Name':<45} | {'Sensitivity':<12} | {'Specificity':<12} | {'Precision':<12} | {'F1-Score':<12} | {'Support':<10}")
    print("-"*120)
    
    for m in metrics:
        if m['support'] > 0:  # Only show classes with samples
            print(f"{m['class_name']:<45} | {m['sensitivity']:.4f}       | {m['specificity']:.4f}       | "
                  f"{m['precision']:.4f}       | {m['f1']:.4f}       | {m['support']:<10}")
    
    print("-"*120)
    
    # Calculate averages
    classes_with_support = [m for m in metrics if m['support'] > 0]
    total_support = sum(m['support'] for m in classes_with_support)
    
    if classes_with_support:
        macro_sens = np.mean([m['sensitivity'] for m in classes_with_support])
        macro_spec = np.mean([m['specificity'] for m in classes_with_support])
        macro_prec = np.mean([m['precision'] for m in classes_with_support])
        macro_f1 = np.mean([m['f1'] for m in classes_with_support])
        
        weighted_sens = sum(m['sensitivity'] * m['support'] for m in classes_with_support) / total_support
        weighted_spec = sum(m['specificity'] * m['support'] for m in classes_with_support) / total_support
        weighted_prec = sum(m['precision'] * m['support'] for m in classes_with_support) / total_support
        weighted_f1 = sum(m['f1'] * m['support'] for m in classes_with_support) / total_support
        
        print(f"{'Macro Average':<45} | {macro_sens:.4f}       | {macro_spec:.4f}       | "
              f"{macro_prec:.4f}       | {macro_f1:.4f}       | {total_support:<10}")
        print(f"{'Weighted Average':<45} | {weighted_sens:.4f}       | {weighted_spec:.4f}       | "
              f"{weighted_prec:.4f}       | {weighted_f1:.4f}       | {total_support:<10}")
    
    print("="*120)
    
    # Detailed breakdown
    print("\n" + "="*120)
    print("DETAILED PER-CLASS BREAKDOWN")
    print("="*120)
    
    for m in metrics:
        if m['support'] == 0:
            continue
            
        print(f"\n{m['class_name']}:")
        print(f"  Support: {m['support']} samples")
        print(f"  True Positives (TP): {m['tp']}")
        print(f"  True Negatives (TN): {m['tn']}")
        print(f"  False Positives (FP): {m['fp']}")
        print(f"  False Negatives (FN): {m['fn']}")
        print(f"  Sensitivity (Recall): {m['sensitivity']:.4f} - Ability to correctly identify this arrhythmia")
        print(f"  Specificity: {m['specificity']:.4f} - Ability to correctly identify non-cases")
        print(f"  Precision: {m['precision']:.4f} - Accuracy when predicting this arrhythmia")
        print(f"  F1-Score: {m['f1']:.4f} - Harmonic mean of precision and recall")
    
    print("\n" + "="*120)


def evaluate():
    """Main evaluation function"""
    print("="*120)
    print("STARTING MODEL EVALUATION")
    print("="*120)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    print("\nLoading dataset from SQL database...")
    try:
        dataset = ECGRawDatasetSQL()
        print(f"✅ Dataset loaded: {len(dataset)} segments")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("❌ No data found in SQL database.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Load Model
    ckpt_path = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"\nLoading model from {ckpt_path}...")
    try:
        model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
        # Use weights_only=False for compatibility with older PyTorch versions
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
            if "epoch" in state:
                print(f"✅ Model trained for {state['epoch']} epochs")
            if "balanced_acc" in state:
                print(f"✅ Best balanced accuracy during training: {state['balanced_acc']:.4f}")
        else:
            model.load_state_dict(state)
            
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run Inference
    y_true = []
    y_pred = []
    
    print("\nRunning inference on all samples...")
    try:
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating", ncols=80):
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                y_pred.extend(preds)
                y_true.extend(y.numpy())
        
        print(f"✅ Inference completed on {len(y_true)} samples")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

    # Calculate metrics
    metrics, cm = calculate_per_class_metrics(y_true, y_pred, len(CLASS_NAMES))
    
    # Print report
    print_report(metrics, cm, len(y_true))
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = BASE_DIR / "models_training" / "outputs" / "logs" / f"evaluation_report_{timestamp}.txt"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print_report(metrics, cm, len(y_true))
            sys.stdout = original_stdout
        
        print(f"\n✅ Evaluation report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save report to file: {e}")
    
    print("\n" + "="*120)
    print("EVALUATION COMPLETED")
    print("="*120)


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
