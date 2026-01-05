
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

# --- IMPORTS ---
from simulation.synthesize_arrhythmias import SyntheticECGGenerator
from signal_processing.cleaning import clean_signal
from signal_processing.artifact_detection import check_signal_quality
from decision_engine.rhythm_orchestrator import RhythmOrchestrator
from xai.xai import explain_segment, _load_model, _init_device

# Mock feature extractor (simplified version of dashboard logic)
# We need this because we can't easily import app.py without side effects
def extract_features_sim(signal, fs):
    # simple peak detection or use simplified logic
    # precise feature extraction is secondary to model logic validation
    # For simulation, we can cheat slightly and use the 'expected_r_peaks' 
    # if we wanted, but let's try to be fair.
    
    # Simple heuristic peak detection
    # Squaring strategy
    try:
        diff_sig = np.diff(signal)
        squared = diff_sig ** 2
        # smooth
        window = int(0.12 * fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        # threshold
        thresh = np.mean(integrated) * 2.0
        peaks = []
        last_peak = -999
        for i in range(len(integrated)):
            if integrated[i] > thresh:
                if i - last_peak > fs * 0.2: # 200ms refractory
                    peaks.append(i)
                    last_peak = i
        
        rr_intervals = (np.diff(peaks) / fs * 1000).tolist() # ms list
        hr = 60 / (np.mean(rr_intervals) / 1000) if len(rr_intervals) > 0 else 0
        
        # Simple QRS width estimation (mock)
        # In simulation, we know the label, so we could cheat, but let's just use defaults
        # or simple width if we had per-beat segmentation.
        # For now, let's provide a list of 80ms (Narrow) or 140ms (Wide) based on simplistic heuristic
        # If signal is 'Wide' (e.g. PVC), we might detect it? 
        # Actually, extracting QRS width from ID-less signal is hard without a real delineator.
        # Let's generate a dummy list matching pulse count.
        qrs_durations = [80] * len(peaks)
        
        return {
            "mean_hr": hr,
            "rr_variability": np.std(rr_intervals)/np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            "qrs_width": 80, 
            "pr_interval": 160,
            "qrs_durations_ms": qrs_durations,
            "rr_intervals": rr_intervals
        }
    except:
        return {"mean_hr": 0, "rr_variability": 0, "qrs_durations_ms": [], "rr_intervals": []}

def run_validation_suite(n_samples_per_class=5):
    print("\n" + "="*50)
    print(">>> AUTOMATED VALIDATION SUITE")
    print("="*50)
    
    # 1. Initialize
    print("DEBUG: init generator...")
    gen = SyntheticECGGenerator(fs=250, duration=10) 
    print("DEBUG: init orchestrator...")
    orch = RhythmOrchestrator()
    print("DEBUG: init device...")
    _init_device()
    try:
        print("DEBUG: loading model...")
        _load_model()
        print("DEBUG: model loaded.")
    except Exception as e:
        print(f"[X] Model load failed: {e}")
        return

    # 2. Define Test Cases (Comprehensive)
    test_classes = [
        "Sinus Rhythm", 
        "Sinus Bradycardia", 
        "Sinus Tachycardia",
        "Atrial Fibrillation", 
        "Atrial Flutter",
        "Junctional Rhythm",
        "Idioventricular Rhythm",
        "Ventricular Tachycardia",
        "Ventricular Fibrillation",
        "1st Degree AV Block",
        "2nd Degree AV Block Type 1",
        "2nd Degree AV Block Type 2", 
        "3rd Degree AV Block",
        "PVC", 
        "PVC Bigeminy",
        "PVC Trigeminy",
        "PAC",
        "Bundle Branch Block",
        "Sinus Bradycardia + PVC",
        "Sinus Tachycardia + PVC"
    ]
    
    results = []
    
    # 3. Validation Loop
    for label in test_classes:
        print(f"\nProcessing {label}...")
        for i in range(n_samples_per_class):
            sig, r_peaks, meta = gen.generate_segment(label)
            clean_sig = clean_signal(sig, 250)
            sqi_res = check_signal_quality(clean_sig, 250)
            feats = extract_features_sim(clean_sig, 250)
            
            try:
                xai_out = explain_segment(clean_sig, feats)
                ml_pred = {
                    "label": xai_out.get("pred_label"),
                    "probs": xai_out.get("probabilities"),
                    "confidence": max(xai_out.get("probabilities", [0])) if xai_out.get("probabilities") else 0
                }
            except Exception as e:
                ml_pred = {"label": "Error", "probs": [], "confidence": 0}
            
            decision = orch.decide(ml_pred, feats, sqi_res)
            
            final_pred = decision["final_label"]
            results.append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "expected": label,
                "predicted": final_pred,
                "sqi_status": "Pass" if sqi_res["is_acceptable"] else "Fail",
                "source": decision["source"]
            })
            
    # 4. Analysis
    df = pd.DataFrame(results)
    
    # Save History
    history_file = BASE_DIR / "validation" / "history.csv"
    if history_file.exists():
        df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        df.to_csv(history_file, index=False)
    print(f"\n[+] Results saved to {history_file}")

    print("\n" + "-"*50)
    print("VALIDATION REPORT")
    print("-"*50)
    
    y_true = df["expected"].tolist()
    y_pred = df["predicted"].tolist()
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.2%}")
    
    print("\nClassification Report:")
    # Calculate uniquely present labels
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, digits=3, labels=unique_labels))
    
    print("\nFailure Cases (Mismatch):")
    mismatches = df[df["expected"] != df["predicted"]]
    if len(mismatches) > 0:
        print(mismatches[["expected", "predicted", "source", "sqi_status"]].to_string())
    else:
        print("None! Perfect Match.")

    # 5. Pass/Fail
    if acc > 0.8:
        print("\n[+] VALIDATION PASSED (Threshold > 80%)")
        sys.exit(0)
    else:
        print("\n[-] VALIDATION FAILED (Threshold > 80%)")
        sys.exit(1)

if __name__ == "__main__":
    run_validation_suite()
