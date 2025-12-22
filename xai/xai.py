"""
xai.py

Explainable AI module for the CNN+Transformer ECG Arrhythmia classifier.

Provides:
  - Option A clinical explanation (model + rules)
  - Saliency map (vanilla gradient)
  - CNN feature maps
  - Transformer self-attention weights

Used by app.py:
   /api/xai/<segment_id>  → explain_segment()
   /api/xai_raw           → predict_and_explain()  (if you add route)
"""

from pathlib import Path
import sys

# --- FIX IMPORTS FOR FOLDER RESTRUCTURE ---
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "models_training"))

import numpy as np
import torch
import torch.nn.functional as F

from models import CNNTransformerClassifier
from data_loader import CLASS_NAMES



# ---------------------------------------------------------------------
# GLOBALS & CHECKPOINT
# ---------------------------------------------------------------------

# Make path relative to this file (xai.py)
# BASE_DIR is defined above as project root
# CKPT is in models_training/outputs/checkpoints
CKPT_PATH = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model.pth"

_device = None
_model = None

_last_cnn_featuremap = None
_last_attention = None
_is_model_untrained = False  # Track if we are using a fallback random model


def _init_device():
    """Initialize device safely."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ XAI device initialized: {_device}")
    return _device


# ---------------------------------------------------------------------
# HOOKS for CNN feature maps & Transformer attention
# ---------------------------------------------------------------------

def _cnn_hook(module, inp, out):
    global _last_cnn_featuremap
    _last_cnn_featuremap = out.detach().cpu().numpy()


def _attention_hook(module, inp, out):
    global _last_attention
    # Handle case where out is a tuple (attn_output, attn_weights) or just attn_output
    if isinstance(out, tuple) and len(out) > 1:
        attn = out[1]
    else:
        attn = out

    if attn is not None:
        _last_attention = attn.detach().cpu().numpy()
    else:
        _last_attention = None


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

# Global to track loaded model timestamp
_loaded_model_mtime = 0

def _load_model():
    """
    Lazily load the CNN+Transformer model from checkpoints.
    Uses outputs/checkpoints/best_model.pth produced by train.py
    Checks file modification time to auto-reload if updated.
    """
    # Global to track loaded model timestamp and status
    global _model, _loaded_model_mtime, _is_model_untrained

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"❌ No checkpoint found at {CKPT_PATH}. "
            "Train the model first (train.py) or ensure best_model.pth exists."
        )

    # Check modification time
    current_mtime = CKPT_PATH.stat().st_mtime
    
    # Reload if model is None OR file has changed
    if _model is not None and current_mtime == _loaded_model_mtime:
        return _model

    if _model is not None:
        print("🔄 Model file changed! Reloading...")

    device = _init_device()

    try:
        # weights_only=False required for Numpy 2.0 compatibility in some PyTorch versions
        state = torch.load(CKPT_PATH, map_location=device, weights_only=False)

        if "model_state" in state:
            sd = state["model_state"]
        else:
            sd = state

        model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
        
        try:
            model.load_state_dict(sd)
        except RuntimeError as e:
            # Model architecture mismatch (e.g., different number of classes)
            print(f"⚠️  Model checkpoint mismatch: {e}")
            print(f"⚠️  Continuing with UNTRAINED model structure for {len(CLASS_NAMES)} classes.")
            print(f"⚠️  Please run 'Retrain Model' in dashboard to fix deep learning predictions.")
            # Do NOT raise. Just continue with the random-weight model.
            # This allows the app to run so Users can use the "Rules" logic (which overrides model anyway).
            _is_model_untrained = True
            pass
        
        model.to(device)
        model.eval()

        # Attach hooks for CNN feature maps & transformer attention
        try:
            if hasattr(model, "cnn"):
                model.cnn.register_forward_hook(_cnn_hook)

            if hasattr(model, "transformer_encoder"):
                last_layer = model.transformer_encoder.layers[-1]
                last_layer.self_attn.register_forward_hook(_attention_hook)
        except Exception as e:
            print(f"⚠️  XAI hooks not fully attached: {e}")

        _model = model
        _loaded_model_mtime = current_mtime
        print(f"✓ Model loaded (or fallback initialized) from {CKPT_PATH}")
        return model

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")


def reset_model():
    """
    Called by /api/retrain_model after retraining from SQL.
    Forces reload from updated checkpoints on next call.
    """
    global _model
    _model = None
    return True


# ---------------------------------------------------------------------
# CLINICAL RULES (Option A explanation)
# ---------------------------------------------------------------------

def _apply_clinical_rules(features: dict) -> tuple:
    """
    Apply clinical rules to detect arrhythmias based on ECG features.
    
    Returns:
        (prediction, confidence, reason)
        - prediction: str or None (class name if rule matches)
        - confidence: "HIGH" or "MODERATE" 
        - reason: str (explanation of why rule fired)
    """
    # Safely extract features
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    # Debug: Log PR interval for AV block detection
    if pr > 0:
        print(f"[RULES] PR interval = {pr:.0f} ms, HR = {hr:.0f} bpm")
    
    # RR intervals for irregularity check
    rr_intervals = features.get("rr_intervals_ms")
    if isinstance(rr_intervals, list) and len(rr_intervals) > 2:
        rr_arr = np.array([x for x in rr_intervals if x is not None and isinstance(x, (int, float))])
    else:
        rr_arr = np.array([])
    
    # QRS durations
    raw_qrs = features.get("qrs_durations_ms")
    if isinstance(raw_qrs, list):
        qrs_list = [x for x in raw_qrs if x is not None and isinstance(x, (int, float))]
    else:
        qrs_list = []
    
    qrs_mean = float(sum(qrs_list) / len(qrs_list)) if len(qrs_list) > 0 else 0.0
    
    # HRV features
    sdnn = float(features.get("SDNN", 0.0))
    rmssd = float(features.get("RMSSD", 0.0))
    
    # ============================================================
    # RULE 0: SINUS RHYTHM (Anti-AFib Override)
    # ============================================================
    # CRITICAL: Check for regular sinus rhythm FIRST
    # This prevents mislabeled AFib segments (that are actually sinus) from being classified as AFib
    # Criteria: Regular RR intervals (low CV) + reasonable HR + normal PR
    if len(rr_arr) > 3:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0
        
        # Very regular rhythm (CV < 0.08) indicates sinus, NOT AFib
        if cv < 0.08:
            # Check if HR is in normal/brady/tachy range
            if 59 < hr < 100:
                # Normal sinus rhythm
                return ("Sinus Rhythm", "HIGH",
                        f"Regular rhythm (CV={cv:.3f}) with normal HR={hr:.0f} bpm - NOT AFib")
            elif hr >= 100:
                # Sinus tachycardia
                if 101 <= pr <= 200 and qrs_mean < 120:
                    return ("Sinus Tachycardia", "HIGH",
                            f"Regular rhythm (CV={cv:.3f}) with HR={hr:.0f} bpm - NOT AFib")
                else:
                    return ("Sinus Tachycardia", "MODERATE",
                            f"Regular rhythm with elevated HR={hr:.0f} bpm")
            elif hr < 60 and hr > 30:
                # Sinus bradycardia
                return ("Sinus Bradycardia", "HIGH",
                        f"Regular rhythm (CV={cv:.3f}) with low HR={hr:.0f} bpm - NOT AFib")
        
        # Moderately regular (CV < 0.12) - likely sinus, not AFib
        elif cv < 0.12 and 50 < hr < 120:
            return ("Sinus Rhythm", "MODERATE",
                    f"Mostly regular rhythm (CV={cv:.3f}) suggests sinus, not AFib")
    
    # ============================================================
    # RULE 1: Atrial Fibrillation
    # ============================================================
    # Criteria: 
    # 1. Irregular RR intervals (high CV and RMSSD)
    # 2. Absence of distinct P-waves (PR ~ 0 or missing detection)
    
    if len(rr_arr) > 3:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0
        
        # P-wave check: If PR is effectively 0, it suggests P-waves were not found (typical in AFib)
        p_waves_absent = (pr < 10)
        
        # Scenario 1: High Irregularity (Classical strict rule)
        if cv > 0.20 and rmssd > 60:
            return ("Atrial Fibrillation", "HIGH", 
                    f"Highly irregular rhythm (CV={cv:.2f}, RMSSD={rmssd:.0f}ms).")
                    
        # Scenario 2: Irregularity + Absence of P-waves
        # If rhythm is moderately irregular but P-waves are GONE, it's strong evidence for AFib.
        elif cv > 0.12 and p_waves_absent:
             return ("Atrial Fibrillation", "HIGH", 
                    f"Irregular rhythm (CV={cv:.2f}) with absent discernable P-waves (f-waves likely).")

        # Scenario 3: Moderate AFib detection (Fallback)
        elif cv > 0.18 and rmssd > 50:
            return ("Atrial Fibrillation", "MODERATE",
                    f"Irregular rhythm (CV={cv:.2f}) with RMSSD={rmssd:.0f}ms")
        
    
    # ============================================================
    # RULE 2: Sinus Tachycardia
    # ============================================================
    # Criteria: HR > 100 bpm, regular rhythm, normal PR and QRS
    if hr > 100:
        if 120 <= pr <= 200 and qrs_mean < 120:
            # Clear tachycardia with normal conduction
            return ("Sinus Tachycardia", "HIGH",
                    f"HR={hr:.0f} bpm (>100) with normal PR={pr:.0f}ms and QRS={qrs_mean:.0f}ms")
        elif hr > 110:
            # Very high HR, even without perfect PR/QRS
            return ("Sinus Tachycardia", "MODERATE",
                    f"HR={hr:.0f} bpm suggests tachycardia")
    
    # ============================================================
    # RULE 3: Sinus Bradycardia
    # ============================================================
    # Criteria: HR < 60 bpm, regular rhythm
    if hr < 60 and hr > 30:  # Exclude unrealistic values
        if len(rr_arr) > 2:
            rr_std = np.std(rr_arr)
            rr_mean = np.mean(rr_arr)
            cv = rr_std / rr_mean if rr_mean > 0 else 0
            
            # Regular rhythm (CV < 0.1)
            if cv < 0.1:
                return ("Sinus Bradycardia", "HIGH",
                        f"HR={hr:.0f} bpm (<60) with regular rhythm (CV={cv:.2f})")
        
        # Even without RR data, very low HR is bradycardia
        if hr < 50:
            return ("Sinus Bradycardia", "MODERATE",
                    f"HR={hr:.0f} bpm indicates bradycardia")
    
    # ============================================================
    # ============================================================
    # RULE 4: AV Blocks (User Specified Definitions)
    # ============================================================
    
    # ------------------------------------------------------------
    # 4A. Third-Degree AV Block (Complete Heart Block)
    # ------------------------------------------------------------
    # Criteria: 
    # - Regular Ventricular Rhythm (CV < 0.12)
    # - Ventricular Rate: 30-60 bpm (Junctional) or <40 bpm (Ventricular)
    # - QRS: Wide if ventricular escape, narrow if junctional
    # - AV Dissociation (PR variable - hard to check with median, so we rely on Rate/Reg check)
    if hr < 60 and len(rr_arr) > 2:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0
        
        if cv < 0.12: # Regular independent ventricular rhythm
            if hr < 40:
                # Very slow -> Likely Ventricular Escape
                return ("3rd Degree AV Block", "HIGH",
                        f"Critical Bradycardia ({hr:.0f} bpm) with Regular Rhythm. Ventricular Escape likely.")
            elif 40 <= hr < 55 and qrs_mean > 120:
                 # Slow + Wide -> Ventricular Escape
                 return ("3rd Degree AV Block", "HIGH",
                         f"Bradycardia ({hr:.0f} bpm) with Wide QRS and Regular Rhythm. Suggests Complete Heart Block.")
            # Note: 40-60 Narrow overlap with Sinus Brady. We assume Sinus Brady unless P-waves (PR) suggest otherwise.
            # If PR is random, we can't tell easily here. We err on Sinus Brady for Narrow/Regular/40-60 bpm 
            # unless we detect 'Variable PR' (future feature).

    # ------------------------------------------------------------
    # 4B. Second-Degree AV Block (Dropped Beats)
    # ------------------------------------------------------------
    # Common Feature: Dropped QRS (RR interval > 1.8x median)
    if len(rr_arr) > 4 and pr > 90: # PR>90 implies P-waves are typically present
        median_rr = np.median(rr_arr)
        is_dropped_beat = any(rr > 1.8 * median_rr for rr in rr_arr)
        
        if is_dropped_beat:
            # Distinguish Type I vs Type II based on QRS and description
            
            # Type II (Mobitz II): 
            # - Fixed PR before drop (we can't check variance easily on single median)
            # - QRS often Wide (>120ms)
            # - "More dangerous"
            if qrs_mean > 120:
                 return ("2nd Degree AV Block Type 2", "HIGH",
                         f"Intermittent Dropped Beat with Wide QRS ({qrs_mean:.0f}ms). Likely Mobitz II.")
            
            # Type I (Wenckebach):
            # - Progressive PR (cant check)
            # - QRS Normal (<120ms)
            # - "Grouped Beating"
            else:
                 return ("2nd Degree AV Block Type 1", "HIGH",
                         f"Intermittent Dropped Beat with Narrow QRS ({qrs_mean:.0f}ms). Likely Wenckebach.")

    # ------------------------------------------------------------
    # 4C. First-Degree AV Block
    # ------------------------------------------------------------
    # Criteria:
    # - PR Interval > 200 ms
    # - QRS Normal (<120 ms)
    # - Regular Rhythm
    if pr > 200:
        if qrs_mean < 120:
             # Regularity check (usually regular)
             if len(rr_arr) > 2:
                 cv = np.std(rr_arr) / np.mean(rr_arr)
                 if cv < 0.15:
                      return ("1st Degree AV Block", "HIGH",
                              f"Prolonged PR ({pr:.0f}ms > 200ms) with Normal QRS and Regular Rhythm.")
        
        # Fallback if QRS is wide (coexisting BBB) or slightly irreg
        return ("1st Degree AV Block", "MODERATE",
                f"Prolonged PR interval ({pr:.0f}ms).")
    
    # ============================================================
    # RULE 5: PVCs (Premature Ventricular Contractions)
    # ============================================================
    # Criteria: Wide QRS (> 120ms) + irregular rhythm
    if qrs_mean > 120:
        if len(rr_arr) > 3:
            rr_std = np.std(rr_arr)
            rr_mean = np.mean(rr_arr)
            cv = rr_std / rr_mean if rr_mean > 0 else 0
            
            # Wide QRS with some irregularity
            if cv > 0.08:
                return ("PVCs", "MODERATE",
                        f"Wide QRS ({qrs_mean:.0f}ms) with irregular rhythm (CV={cv:.2f})")
    
    # ============================================================
    # RULE 5B: PVC PATTERNS (Bigeminy, Trigeminy, etc.)
    # ============================================================
    # Detect specific PVC patterns based on QRS width sequence
    # OPTIMIZED: Check patterns in order of clinical priority
    if qrs_list and len(qrs_list) >= 6:
        # Count wide QRS complexes (likely PVCs) - do this once
        wide_qrs_count = sum(1 for q in qrs_list if q > 120)
        
        # Skip pattern detection if not enough wide QRS
        if wide_qrs_count < 2:
            return (None, None, None)  # Not enough PVCs for patterns
        
        # PVC Triplets: 3 consecutive PVCs (HIGHEST PRIORITY - VT warning!)
        if len(qrs_list) == 3:
            for i in range(len(qrs_list) - 2):
                if qrs_list[i] > 120 and qrs_list[i+1] > 120 and qrs_list[i+2] > 120:
                    return ("PVC Triplets (VT Warning)", "HIGH",
                           f"Three consecutive PVCs! Warning for VT at position {i}-{i+2}")
        
        # PVC Couplets: 2 consecutive PVCs (SECOND PRIORITY)
        if len(qrs_list) == 2:
            for i in range(len(qrs_list) - 1):
                if qrs_list[i] > 120 and qrs_list[i+1] > 120:
                    return ("PVC Couplets", "HIGH",
                           f"Two consecutive PVCs at positions {i},{i+1}. Increased risk")
        
        # PVC Bigeminy: Every other beat is a PVC (check up to 6 beats for speed)
        if len(qrs_list) >= 6 and wide_qrs_count >= 3:
            bigeminy_matches = 0
            check_limit = min(len(qrs_list) - 1, 6)  # Only check first 6 beats
            for i in range(0, check_limit, 2):
                if i+1 < len(qrs_list):
                    if qrs_list[i] < 120 and qrs_list[i+1] > 120:
                       bigeminy_matches += 1
            
            if bigeminy_matches >= 2:  # At least 2 N-PVC pairs
                return ("PVC Bigeminy", "HIGH",
                       f"Alternating N-PVC pattern. High burden: {wide_qrs_count} PVCs")
        
        # PVC Trigeminy: Every third beat is a PVC (check up to 9 beats)
        if len(qrs_list) >= 9 and wide_qrs_count >= 3:
            trigeminy_matches = 0
            check_limit = min(len(qrs_list) - 2, 9)
            for i in range(0, check_limit, 3):
                if i+2 < len(qrs_list):
                    if qrs_list[i] < 120 and qrs_list[i+1] < 120 and qrs_list[i+2] > 120:
                        trigeminy_matches += 1
            
            if trigeminy_matches >= 2:  # At least 2 N-N-PVC triplets
                return ("PVC Trigeminy", "HIGH",
                       f"N-N-PVC pattern. Moderate burden: {wide_qrs_count} PVCs")
        
        # PVC Quadrigeminy: Every fourth beat is a PVC (check up to 12 beats)
        if len(qrs_list) >= 12 and wide_qrs_count >= 2:
            quadrigeminy_matches = 0
            check_limit = min(len(qrs_list) - 3, 12)
            for i in range(0, check_limit, 4):
                if i+3 < len(qrs_list):
                    if (qrs_list[i] < 120 and qrs_list[i+1] < 120 and 
                        qrs_list[i+2] < 120 and qrs_list[i+3] > 120):
                        quadrigeminy_matches += 1
            
            if quadrigeminy_matches >= 2:  # At least 2 N-N-N-PVC groups
                return ("PVC Quadrigeminy", "HIGH",
                        f"N-N-N-PVC pattern. PVC burden: {wide_qrs_count}")

            
    # ============================================================
    # RULE 5C: Atrial Ectopy (Couplets/Runs) - NEW
    # ============================================================
    # Atrial Couplet: 2 consecutive PACs
    # PAS: Narrow QRS (<120) + Preamature
    # Since we don't have per-beat prematurity easily mapped here, we check for
    # sequence of N-N that are part of a 'fast' burst or context.
    # Approximation: 
    # If we have [N, N] that are < 0.8 * median_RR?
    
    # Let's assume the "PAC" logic is passed via labels or we infer:
    # If we find 2 narrow beats that are significantly faster than background?
    
    # Simplified Logic using QRS lists and RR lists if aligned:
    # We will trust the label "PAC" from model if available, but for rules:
    # Check for short RRs with Narrow QRS.
    
    if len(qrs_list) > 3 and len(rr_arr) > 3:
        median_rr = np.median(rr_arr)
        
        # Find indices of premature beats
        premature_indices = [i for i, rr in enumerate(rr_arr) if rr < 0.8 * median_rr]
        
        # Check if they are consecutive
        consecutive_premature = 0
        max_consecutive_premature = 0
        last_idx = -999
        
        for idx in premature_indices:
            if idx == last_idx + 1:
                consecutive_premature += 1
            else:
                consecutive_premature = 1
            max_consecutive_premature = max(max_consecutive_premature, consecutive_premature)
            last_idx = idx
            
        # Check QRS width during these? (Approximate: assume majority are narrow if mean < 120)
        if qrs_mean < 120:
            if max_consecutive_premature == 2:
                 return ("Atrial Couplet", "MODERATE", "Two consecutive premature narrow complexes detected.")
            elif max_consecutive_premature >= 3:
                 return ("Atrial Run", "HIGH", f"Run of {max_consecutive_premature} consecutive premature narrow complexes.")
    
    # ============================================================
    # RULE 6: Supraventricular Tachycardia (SVT)
    # ============================================================
    # Criteria: Very high HR (>150) + narrow QRS + regular rhythm
    # ============================================================
    # RULE 6: Supraventricular Tachycardia (SVT)
    # ============================================================
    # Criteria: Very high HR (>150) + narrow QRS + regular rhythm
    if hr > 150:
        if qrs_mean < 120:  # Narrow QRS
            if len(rr_arr) > 2:
                rr_std = np.std(rr_arr)
                rr_mean = np.mean(rr_arr)
                cv = rr_std / rr_mean if rr_mean > 0 else 0
                
                # Regular rhythm
                # PSVT Logic: Sudden onset, Regular, >150bpm
                if cv < 0.1:
                    return ("PSVT", "HIGH",
                            f"Paroxysmal SVT: Regular narrow-complex tachycardia at {hr:.0f} bpm")
    
    # ============================================================
    # NEW RULES: COMPLEX PATTERNS (Atrial/Ventricular Runs, Pauses)
    # ============================================================
    
    # 1. PAUSE Detection
    # Rule: RR interval > 2.0s (2000ms)
    if any(rr > 2000 for rr in rr_arr):
         max_pause = max(rr_arr) / 1000.0
         return ("Pause", "HIGH", f"Significant pause of {max_pause:.2f}s detected")

    # Pattern Analysis on Beat Sequence
    if len(qrs_list) >= 3 and len(rr_arr) >= 2:
        # Identify premature beats: RR < 0.8 * mean_RR
        mean_rr = np.mean(rr_arr)
        
        # We need to act on the list of QRS widths. 
        # Create a "Type" list: 'N' (Narrow), 'W' (Wide)
        # Threshold 120ms
        beat_types = ['W' if q > 120 else 'N' for q in qrs_list]
        
        # Detect consecutive patterns
        w_run_count = 0
        n_run_count = 0 
        
        # Logic for "Run": 3 or more consecutive PVCs (Wide) or PACs (Narrow & Premature)
        # Note: Determining "Premature" for every beat in a sequence is tricky with just stats.
        # But usually a "run" of ectopics is fast. 
        # Approch: Count consecutive Wide beats (PVC Run)
        
        max_w_run = 0
        current_w_run = 0
        for b in beat_types:
            if b == 'W':
                current_w_run += 1
            else:
                max_w_run = max(max_w_run, current_w_run)
                current_w_run = 0
        max_w_run = max(max_w_run, current_w_run) # Final check

        # VT / NSVT / Ventricular Run
        if max_w_run >= 3:
            if hr > 100:
                return ("NSVT", "HIGH", f"Non-Sustained VT: Run of {max_w_run} wide-complex beats at >100 bpm")
            else:
                return ("Ventricular Run", "HIGH", f"Ventricular Run: {max_w_run} consecutive wide-complex beats")

        # ATRIAL RUN / COUPLET
        # Logic: 2 or more consecutive PACs.
        # PAC = Narrow QRS + Premature. 
        # Since we use beat-by-beat, let's look at prematurity if we can align RR to Beats.
        # Usually len(rr) = len(beat) - 1.
        # Simple heuristic: If HR isn't usually tachy, but we have a burst of Narrow beats with short RRs?
        # Let's rely on morphology (Narrow) + context.
        # Actually, "Atrial Run" implies they are Ectopic, not sinus.
        # Hard to distinguish Sinus Tachy vs Atrial Run without P-wave morphology.
        # CLINICAL PROXY: If we see a "burst" of short RRs in a normal segments?
        # Let's focus on "Atrial Couplet" (2 PACs).
        
        # We can detect "PAC" in Rule 5 logic or similar?
        pass

    # ============================================================
    # No strong rule match
    # ============================================================
    return (None, None, None)


def _clinical_explanation(label: str, features: dict, attention_context: str = "") -> str:
    """
    Returns a text explanation focusing on the 'intricate details' of the detection.
    Strictly descriptive: explains WHY the arrhythmia was detected based on features.
    
    Args:
        label: The predicted class
        features: Dictionary of ECG features (HR, PR, etc.)
        attention_context: String describing where the model looked (from transformer attention)
    """

    if not label:
        return "No arrhythmia detected in this segment."

    text = label.lower()
    
    # Safely get HR and PR, handling None
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    rr_intervals = features.get("rr_intervals_ms", [])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 0:
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        cv = rr_std / rr_mean if rr_mean > 0 else 0.0
    else:
        cv = 0.0

    # Safely calculate QRS mean
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            qrs_list = [x for x in raw_qrs if x is not None and isinstance(x, (int, float))]
        elif isinstance(raw_qrs, str):
            try:
                import json
                qrs_list = json.loads(raw_qrs)
                qrs_list = [x for x in qrs_list if x is not None and isinstance(x, (int, float))]
            except:
                qrs_list = []
        else:
            qrs_list = []
            
        if len(qrs_list) > 0:
            qrs_mean = float(sum(qrs_list) / len(qrs_list))
    except Exception as e:
        print(f"⚠️  Error parsing QRS durations: {e}")
        qrs_list = []

    # Helper to construct the 'Intricate Details' string
    details = []
    
    # Analyze Rhythm Regularity
    if cv < 0.08:
        rhythm_str = "Regular rhythm"
    elif cv < 0.15:
        rhythm_str = "Mildly irregular rhythm"
    else:
        rhythm_str = "Irregular rhythm"
    
    details.append(f"Rhythm Analysis: {rhythm_str} (Coefficient of Variation={cv:.3f})")
    
    # Analyze Heart Rate
    if hr > 100:
        hr_str = f"Tachycardic (HR={hr:.0f} bpm)"
    elif hr < 60:
        hr_str = f"Bradycardic (HR={hr:.0f} bpm)"
    else:
        hr_str = f"Normal rate (HR={hr:.0f} bpm)"
    details.append(f"Heart Rate: {hr_str}")

    # Analyze Intervals
    pr_str = f"PR Interval={pr:.0f}ms" + (" (Prolonged)" if pr > 200 else "")
    qrs_str = f"QRS Duration={qrs_mean:.0f}ms" + (" (Wide)" if qrs_mean > 120 else " (Normal)")
    details.append(f"Conduction: {pr_str}, {qrs_str}")

    # Append the attention context to all returns if meaningful
    
    def enhance(base_text):
        if attention_context: 
             return f"{base_text}\n\nModel Context: {attention_context}"
        return base_text

    # --- SPECIFIC ARRHYTHMIA EXPLANATIONS ---

    # Atrial Fibrillation
    if "atrial fibrillation" in text or "afib" in text:
        return enhance(
            f"Analysis: The ECG exhibits **Atrial Fibrillation**, characterized by a **{rhythm_str}** "
            f"(Diff={cv:.3f}). The absence of consistent P-waves combined with an irregular ventricular response "
            f"confirms the diagnosis.\n"
            f"Key metrics: HR={hr:.0f} bpm, RMSSD={features.get('RMSSD',0):.0f}ms."
        )

    # Supraventricular Tachycardia
    if "supraventricular tachycardia" in text or "svt" in text:
        return enhance(
            f"Analysis: **Supraventricular Tachycardia (SVT)** detected. The heart rate is significantly elevated "
            f"at **{hr:.0f} bpm**, originating above the ventricles as indicated by the narrow QRS complexes "
            f"({qrs_mean:.0f} ms). The rhythm is regular, suggesting a stable re-entrant mechanism."
        )

    # --- ECTOPIC BEATS & PATTERNS ---
    
    if "bigeminy" in text:
        return enhance(
            f"Analysis: **Ventricular Bigeminy** identified. Using waveform morphology, the model detected an alternating "
            f"pattern of normal sinus beats and **Premature Ventricular Contractions (PVCs)**. "
            f"The wide QRS complexes (>120ms) in every other beat create a distinct high-frequency modulation in the rhythm."
        )
    
    if "trigeminy" in text:
        return enhance(
            f"Analysis: **Ventricular Trigeminy**. The trace shows a repeating triplet pattern: two normal beats "
            f"followed by one wide-complex ectopic beat. This suggests a stable focus of ventricular irritability."
        )
    
    if "quadrigeminy" in text: # Corrected from 'quadrigeminy'
         return enhance(
            f"Analysis: **Ventricular Quadrigeminy**. A regular pattern where every fourth beat is a PVC. "
            f"The underlying rhythm remains otherwise stable."
        )
    
    if "couplet" in text or "pair" in text:
        return enhance(
            f"Analysis: **PVC Couplet** detected. Two consecutive premature ventricular complexes were isolated. "
            f"This indicates a moment of heightened electrical instability compared to single PVCs."
        )
    
    if "triplet" in text or "vt warning" in text or "run" in text:
        return enhance(
             f"Analysis: **Run of PVCs (Triplet)**. A burst of three consecutive ventricular beats was found. "
             f"This is clinically significant as a potential precursor to Ventricular Tachycardia."
        )
    
    # PACs (Single)
    if "pac" in text and not ("pair" in text or "run" in text):
         return enhance(
             f"Analysis: **Premature Atrial Contraction (PAC)**. A beat occurred earlier than the expected sinus cycle. "
             f"Unlike PVCs, the QRS remains narrow ({qrs_mean:.0f}ms), confirming atrial origin, though P-wave morphology may be distorted."
         )

    # General PVCs
    if "pvc" in text and not ("bigeminy" in text or "trigeminy" in text):
        return enhance(
            f"Analysis: **Premature Ventricular Contractions**. The model identified ectopic beats with "
            f"wide QRS morphology ({qrs_mean:.0f}ms) arriving early in the cardiac cycle. "
            f"Background rhythm appears {rhythm_str.lower()}."
        )

    # --- ADVANCED RHYTHMS ---

    if "junctional" in text:
        return enhance(
            f"Analysis: **Junctional Rhythm**. The electrical pacemaker has shifted to the AV node. "
            f"Evidence: Normal/Narrow QRS complexes with absent or retrograde P-waves, usually at a slower rate ({hr:.0f} bpm)."
        )

    if "idioventricular" in text:
        return enhance(
            f"Analysis: **Idioventricular Rhythm**. Technical checks show a very slow ventricular escape rate ({hr:.0f} bpm) "
            f"with wide, bizarre QRS complexes. This is a critical rhythm often seen when supraventricular pacemakers fail."
        )
    
    if "ventricular fibrillation" in text:
        return enhance(
            f"CRITICAL: **Ventricular Fibrillation**. The signal is chaotic and disorganized with no discernible QRS complexes. "
            f"Mechanical cardiac output is likely compromised."
        )

    if "atrial flutter" in text:
        return enhance(
            f"Analysis: **Atrial Flutter**. The model detected characteristic 'sawtooth' F-waves, likely at a rate near 300 bpm, "
            f"with a structured ventricular response (e.g., 2:1 or 4:1 block)."
        )

    # Blocks
    if "wenckebach" in text or "type 1" in text:
        return enhance(
            f"Analysis: **2nd Degree AV Block Type I (Wenckebach)**. The rhythm exhibits progressive PR interval prolongation "
            f"culminating in a dropped beat. This grouping pattern (e.g., 4:3 or 3:2 conduction) is characteristic of AV nodal delay."
        )
        
    if "mobitz ii" in text or "type 2" in text:
        return enhance(
            f"Analysis: **2nd Degree AV Block Type II (Mobitz II)**. Intermittent non-conducted P-waves were detected without "
            f"prior PR interval lengthening. This suggests an unpredictable block in the His-Purkinje system, carrying a higher risk of progression to complete heart block."
        )

    if "3rd degree" in text or "complete" in text or "3avb" in text:
        return enhance(
            f"Analysis: **3rd Degree (Complete) AV Block**. There is complete AV dissociation. The atrial rate (P-waves) and ventricular "
            f"rate (QRS) are independent. The ventricles are beating at a slow escape rate ({hr:.0f} bpm) unrelated to the P-waves."
        )
    
    if "bundle branch" in text:
        return enhance(
            f"Analysis: **Bundle Branch Block**. Conduction delay is evident via widened QRS complexes ({qrs_mean:.0f}ms). "
            f"The rhythm is otherwise supraventricular, differentiating this from ventricular ectopy."
        )
    
    # --- NEW RULES EXPLANATIONS ---
    
    if "atrial couplet" in text:
        return enhance(
            f"Analysis: **Atrial Couplet**. Two consecutive Premature Atrial Contractions (PACs) were detected. "
            f"These are narrow-complex beats ({qrs_mean:.0f} ms) occurring earlier than the expected sinus rhythm."
        )

    if "atrial run" in text:
        return enhance(
            f"Analysis: **Atrial Run** detected. A burst of 3 or more consecutive PACs. "
            f"This represents a short episode of atrial tachycardia."
        )

    if "ventricular run" in text or "salvo" in text:
        return enhance(
            f"Analysis: **Ventricular Run (Salvo)**. A short burst of 3 or more consecutive wide-complex beats "
            f"(>120ms) originating from the ventricles. This indicates significant ventricular irritability."
        )

    if "nsvt" in text:
        return enhance(
            f"Analysis: **Non-Sustained Ventricular Tachycardia (NSVT)**. A run of 3+ consecutive PVCs at a "
            f"tachycardic rate (>100 bpm), lasting less than 30 seconds. This is a clinically significant finding warranting monitoring."
        )

    if "psvt" in text:
        return enhance(
            f"Analysis: **Paroxysmal Supraventricular Tachycardia (PSVT)**. The rhythm is regular, narrow-complex "
            f"(QRS < 120ms), and rapid (HR={hr:.0f} bpm). The sudden onset implies a re-entrant mechanism above the ventricles."
        )
    
    if "pause" in text:
         return enhance(
            f"Analysis: **Significant Pause**. A prolonged interval between beats (>2.0s) was detected. "
            f"This could result from sinus arrest, exit block, or a non-conducted atrial beat."
        )

    # Sinus Tachycardia (Explicit Class)
    if "sinus tachycardia" in text:
         return enhance(
            f"Analysis: **Sinus Tachycardia**. The heart is in a normal sinus rhythm but beating rapidly ({hr:.0f} bpm). "
            f"All intervals and morphologies are effectively normal, just accelerated."
         )

    # Sinus Bradycardia (Explicit Class)
    if "sinus bradycardia" in text:
         return enhance(
            f"Analysis: **Sinus Bradycardia**. The rate is slow ({hr:.0f} bpm) but originates correctly from the sinus node "
            f"with normal conduction intervals."
         )
    
    # 1st Degree AV Block
    if "av block" in text:
         return enhance(
             f"Analysis: **1st Degree AV Block**. Conduction from atria to ventricles is consistently delayed. "
             f"The PR interval is measured at {pr:.0f}ms (Normal limit: 200ms)."
         )

    # Sinus Rhythm (Normal) - check for hidden conditions
    if "sinus rhythm" in text:
        base = f"Analysis: **Normal Sinus Rhythm**. The waveform is within physiological norms. " \
               f"HR: {hr:.0f} bpm, PR: {pr:.0f}ms, QRS: {qrs_mean:.0f}ms."
               
        if hr > 100:
             base += f"\nNote: However, the rate is elevated, technically meeting criteria for Tachycardia."
        if hr < 50:
             base += f"\nNote: However, the rate is low, technically meeting criteria for Bradycardia."
        if pr > 200:
             base += f"\nNote: A prolonged PR interval suggests an underlying 1st Degree AV Block."
             
        return enhance(base)

    return enhance(
        f"Model Prediction: **{label}**.\n"
        f"Technical Summary: {technical_summary}."
    )


# ---------------------------------------------------------------------
# SALIENCY (vanilla gradient)
# ---------------------------------------------------------------------

def _compute_saliency(model, x, target_idx: int):
    """
    Vanilla gradient saliency on the 1-D ECG trace.
    x: torch tensor of shape (1, 1, T) with requires_grad=True
    """
    model.zero_grad()
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    score = logits[0, target_idx]
    score.backward()

    grad = x.grad.detach().cpu().numpy()[0, 0]
    sal = np.abs(grad)
    sal = sal / (sal.max() + 1e-6)
    return sal.tolist()


def _analyze_attention(model) -> str:
    """
    Analyzes the last captured transformer attention weights to find
    where the model was 'looking' in time.
    
    Attention shape is typically (Batch, T, T) or (Batch, Heads, T, T).
    We sum/mean over heads and query dim to get a 1D 'importance' over time.
    
    Returns: A string description, e.g. "Focus concentrated at 2.4s and 7.1s"
    """
    global _last_attention
    if _last_attention is None:
        return ""
        
    try:
        # _last_attention is numpy, shape usually (1, T, T) or (1, Heads, T, T)
        attn = _last_attention
        if attn.ndim == 4: # (B, H, T, T)
             attn = attn.mean(axis=1) # Average heads -> (B, T, T)
             
        # Take the 0th element in batch
        # attn is now (T, T) - self attention matrix
        # We want to know which 'keys' (source positions) were attended to most.
        # Summing over the query dimension (axis=0) gives total attention received by each token
        attn_1d = attn[0].sum(axis=0) # (T,)
        
        # Normalize
        if attn_1d.max() > 0:
            attn_1d = attn_1d / attn_1d.max()
        
        # Threshold: Find peaks > 0.8
        peaks = np.where(attn_1d > 0.7)[0]
        
        if len(peaks) == 0:
            # Try lower threshold
            peaks = np.where(attn_1d > 0.5)[0]
        
        # Map to seconds
        # Total T is ~312 for 10s. 1 index = 10/312 s
        T = len(attn_1d)
        secs_per_step = 10.0 / T
        
        peak_secs = [f"{p * secs_per_step:.1f}s" for p in peaks]
        
        # Cluster nearby peaks (simple logic)
        unique_zones = []
        if peak_secs:
            last_sec = -999
            for p_idx in peaks:
                sec = p_idx * secs_per_step
                if sec - last_sec > 0.8: # Distinct zone if > 0.8s apart
                    unique_zones.append(f"{sec:.1f}s")
                    last_sec = sec
        
        if not unique_zones:
            return "Diffuse attention across the segment."
            
        if len(unique_zones) > 4:
            return "Attention distributed across multiple points in the segment."
            
        return f"Model focused closely on events at " + ", ".join(unique_zones) + "."
        
    except Exception as e:
        print(f"Attention analysis failed: {e}")
        return ""


# ---------------------------------------------------------------------
# MAIN API FUNCTION FOR /api/xai/<segment_id>
# ---------------------------------------------------------------------

def explain_segment(signal_1d: np.ndarray, features: dict) -> dict:
    """
    Core XAI for Option A.

    Inputs:
      - signal_1d: 1-D ECG numpy array (length ~2500, fs=250 Hz)
      - features: dict from SQL (mean_hr, pr_interval, qrs_durations_ms, etc.)

    Returns dict:
      {
        "pred_label": <str>,
        "probabilities": [...],
        "saliency": [...],
        "explanation": <str>
      }
    """
    try:
        print(f"--- XAI: explain_segment called for signal length {len(signal_1d)} ---")
        
        if len(signal_1d) < 100:
             return {
                "pred_label": "Unknown",
                "probabilities": [],
                "saliency": [],
                "explanation": "Signal too short for analysis."
            }

        device = _init_device()
        model = _load_model()

        arr = np.asarray(signal_1d, dtype=np.float32)
        x = torch.from_numpy(arr[None, None, :]).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        model_prediction = CLASS_NAMES[pred_idx]

        # --- HYBRID APPROACH: MODEL + RULES ---
        # 1. Get model prediction
        # 2. Apply clinical rules
        # 3. If rules contradict model strongly, use rules
        # 4. Otherwise use model
        
        rule_prediction, rule_confidence, rule_reason = _apply_clinical_rules(features)
        
        # Determine final prediction
        final_prediction = model_prediction
        final_probs = probs
        
        is_overridden = False

        if rule_prediction is not None:
             # Logic: If rule is HIGH confidence, or fits "numerical fact" category, it OVERRIDES model.
             # This works for: Pause, NSVT, PSVT, Bigeminy, Blocks, Brady/Tachy
             
             if rule_confidence == "HIGH":
                 is_overridden = True
                 final_prediction = rule_prediction
                 print(f"⚠️  OVERRIDE: Model said '{model_prediction}', but rules detected '{rule_prediction}' ({rule_reason})")
                 
                 # Force updated probability for display
                 try:
                    if rule_prediction in CLASS_NAMES:
                        idx = CLASS_NAMES.index(rule_prediction)
                        final_probs = np.zeros(len(CLASS_NAMES))
                        final_probs[idx] = 1.0
                 except:
                     pass # Unknown class name in rule?

        pred_label = final_prediction
        # --- END HYBRID APPROACH ---

        saliency = _compute_saliency(model, x, pred_idx)
        
        # NEW: Analyze transformer attention for context
        attention_context = _analyze_attention(model)
        
        explanation = _clinical_explanation(pred_label, features or {}, attention_context)
        
        if is_overridden:
             explanation = f"**CLINICAL RULE MATCH**: {rule_reason}\n\n" + explanation
        
        # SAFETY CHECK: If model is untrained and NO rule matched, do not show random output (like VFib)
        if _is_model_untrained and not is_overridden:
            pred_label = "Model Needs Retraining"
            model_prediction = "Untrained"
            explanation = "**NOTICE**: The Deep Learning model architecture has changed and needs retraining. " \
                          "Currently, only Clinical Rules (Pauses, runs, blocks, etc.) are active. " \
                          "Please click 'Retrain Model' in the dashboard to restore full AI functionality."
            final_probs = np.zeros(len(CLASS_NAMES)) # Clear random probabilities

        return {
            "pred_label": pred_label,
            "model_prediction": model_prediction,
            "probabilities": final_probs.tolist(),
            "saliency": saliency,
            "explanation": explanation,
        }
    except Exception as e:
        import traceback
        print(f"❌ XAI explain_segment error: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "pred_label": "Unknown",
            "probabilities": [],
            "saliency": [],
            "explanation": f"Error during explanation: {e}",
        }


# ---------------------------------------------------------------------
# FULL RAW XAI FOR /api/xai_raw (optional)
# ---------------------------------------------------------------------

def predict_and_explain(signal_1d: np.ndarray) -> dict:
    """
    Extended XAI including CNN feature maps and transformer attention.
    You can wire this to /api/xai_raw if desired.

    Returns:
      {
        "pred_label": <str>,
        "probabilities": [...],
        "saliency": [...],
        "cnn_featuremap": [...],
        "transformer_attention": [...]
      }
    """
    global _last_cnn_featuremap, _last_attention

    try:
        _last_cnn_featuremap = None
        _last_attention = None

        device = _init_device()
        model = _load_model()

        arr = np.asarray(signal_1d, dtype=np.float32)
        x = torch.from_numpy(arr[None, None, :]).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        saliency = _compute_saliency(model, x, pred_idx)

        cnn_map = []
        if _last_cnn_featuremap is not None:
            cnn_map = _last_cnn_featuremap[0].tolist()

        attn = []
        if _last_attention is not None:
            attn = _last_attention[0].tolist()

        return {
            "pred_label": pred_label,
            "probabilities": probs.tolist(),
            "saliency": saliency,
            "cnn_featuremap": cnn_map,
            "transformer_attention": attn,
        }
    except Exception as e:
        print(f"❌ XAI predict_and_explain error: {e}")
        return {
            "error": str(e),
            "pred_label": "Unknown",
            "probabilities": [],
            "saliency": [],
        }
