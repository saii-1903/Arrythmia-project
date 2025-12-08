# utils.py
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

CLASS_NAMES = [
    "Sinus Rhythm","Sinus Bradycardia","Sinus Tachycardia","Supraventricular Tachycardia",
    "Ventricular Tachycardia","Atrial Fibrillation","PVCs","Ventricular Bigeminy",
    "Ventricular Trigeminy","1st Degree AV Block","2nd Degree AV Block Type 1",
    "2nd Degree AV Block Type 2","3rd Degree AV Block"
]

def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true)>0 else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average='macro')) if len(y_true)>0 else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm.tolist()}

def save_plot_prediction(sig, probs, outpath):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,3))
    plt.plot(sig, color='k', linewidth=0.8)
    topk = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
    text = "\n".join([f"{CLASS_NAMES[i]}: {p:.2f}" for i,p in topk])
    plt.title(text)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
