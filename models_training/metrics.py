
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def calculate_metrics_report(y_true, y_pred, class_names):
    """
    Generates a formal clinical evaluation report.
    Computes Per-Class Sensitivity (Recall) and Specificity.
    """
    
    # 1. Standard Report (Precision/Recall/F1)
    # Note: Recall = Sensitivity
    cr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 2. Confusion Matrix for Specificity
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # 3. Calculate Specificity per class
    # Specificity = TN / (TN + FP)
    specificities = {}
    
    for i, cls in enumerate(class_names):
        # One-vs-All approach
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Same as recall
        
        specificities[cls] = spec
        
        # Inject into CR dict if needed, or just print separate
    
    # Limit floats
    return cr, specificities

def print_detailed_report(y_true, y_pred, class_names):
    cr, specs = calculate_metrics_report(y_true, y_pred, class_names)
    
    print("\n" + "="*60)
    print("CLINICAL PERFORMANCE METRICS")
    print("="*60)
    print(f"{'Class':<30} | {'Sens (Recall)':<12} | {'Spec':<12} | {'Prec':<12} | {'F1':<12}")
    print("-" * 85)
    
    for cls in class_names:
        if cls not in cr: continue
        
        sens = cr[cls]['recall']
        prec = cr[cls]['precision']
        f1 = cr[cls]['f1-score']
        spec = specs.get(cls, 0.0)
        
        print(f"{cls:<30} | {sens:.3f}        | {spec:.3f}        | {prec:.3f}        | {f1:.3f}")
        
    print("-" * 85)
    print(f"Overall Accuracy: {cr['accuracy']:.3f}\n")
