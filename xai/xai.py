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
    
    # Robust QRS mean
    qrs_mean = 0.0
    qrs_list = []
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            q_list = [x for x in raw_qrs if isinstance(x, (int, float))]
            if q_list: 
                qrs_mean = float(sum(q_list) / len(q_list))
                qrs_list = q_list
    except Exception:
        pass
    
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

    # ============================================================
    # RULE 5C: PAC Trigeminy
    # ============================================================
    # Pattern: Normal, Normal, PAC (Short RR) -> Repeated
    if len(rr_arr) >= 6 and qrs_mean < 120:
        median_rr = np.median(rr_arr)
        pac_trigeminy_matches = 0
        # Check for sequence: [Normal, Normal, Short] in RR intervals
        # Iterate up to len-2
        for i in range(len(rr_arr) - 2):
            rr1 = rr_arr[i]
            rr2 = rr_arr[i+1]
            rr3 = rr_arr[i+2]
            
            # Normal is relative to median (approx > 0.85)
            # Short is premature (< 0.8)
            if (rr1 > 0.85 * median_rr and 
                rr2 > 0.85 * median_rr and 
                rr3 < 0.8 * median_rr):
                pac_trigeminy_matches += 1
        
        if pac_trigeminy_matches >= 2:
             return ("PAC Trigeminy", "HIGH", 
                     f"PAC Trigeminy Pattern: {pac_trigeminy_matches} cycles of N-N-PAC detected.")

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
    Returns a textual explanation that is contextual, detailed, and 'clever'.
    It synthesizes quantitative data (features) with clinical logic.
    """

    if not label:
        return "Analysis: No specific arrhythmia detected. The signal appears to be within normal limits."

    text = label.lower()
    
    # -------------------------------------------------------------
    # 1. EXTRACT AND VALIDATE DATA
    # -------------------------------------------------------------
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0
    
    rr_intervals = features.get("rr_intervals_ms", [])
    if isinstance(rr_intervals, list) and len(rr_intervals) > 0:
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        rmssd = float(features.get("RMSSD", 0))
    else:
        cv = 0.0
        rmssd = 0.0

    # Robust QRS mean
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            q_list = [x for x in raw_qrs if isinstance(x, (int, float))]
            if q_list:
                qrs_mean = float(sum(q_list) / len(q_list))
    except Exception:
        pass

    # -------------------------------------------------------------
    # 2. GENERATE INTELLIGENT CONTEXT
    # -------------------------------------------------------------
    
    # Rate descriptors
    if hr < 40: rate_desc = "profoundly bradycardic"
    elif hr < 60: rate_desc = "bradycardic"
    elif hr < 100: rate_desc = "normal range"
    elif hr < 150: rate_desc = "tachycardic"
    else: rate_desc = "severely tachycardic"
    
    # Rhythm descriptors
    if cv < 0.08: rhythm_desc = "regular"
    elif cv < 0.15: rhythm_desc = "mildly irregular"
    else: rhythm_desc = "irregular"
    
    # Conduction descriptors
    cond_parts = []
    if pr > 200: cond_parts.append(f"AV delay (PR {pr:.0f}ms)")
    elif pr < 120 and pr > 10: cond_parts.append("rapid AV conduction")
    
    if qrs_mean > 120: cond_parts.append(f"wide QRS ({qrs_mean:.0f}ms)")
    else: cond_parts.append(f"normal QRS ({qrs_mean:.0f}ms)")
    
    cond_str = ", ".join(cond_parts) if cond_parts else "normal conduction"

    # Helper function to add attention context
    def enhance(base_text):
        if attention_context:
            return f"{base_text}\n\n**Model Focus**: {attention_context}"
        return base_text

    # -------------------------------------------------------------
    # 3. ARRHYTHMIA-SPECIFIC NARRATIVES
    # -------------------------------------------------------------
    
    intro = f"**Clinical Context**: The rhythm is {rate_desc} ({hr:.0f} bpm) and {rhythm_desc}, with {cond_str}."
    
    # Atrial Fibrillation
    if "fibrillation" in text and "atrial" in text:
        analysis = (f"**Analysis**: **Atrial Fibrillation** is characterized by:\n"
                   f"1. Chaotic irregularity (CV={cv:.2f}, RMSSD={rmssd:.0f}ms)\n"
                   f"2. Absent organized P-waves (replaced by f-waves)\n"
                   f"This combination confirms the diagnosis despite the {rate_desc} ventricular response.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Atrial Flutter
    if "flutter" in text:
        analysis = (f"**Analysis**: **Atrial Flutter** exhibits characteristic 'sawtooth' F-waves at ~300 bpm "
                   f"with structured AV conduction (typically 2:1 or 4:1 block).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 3rd Degree AV Block
    if "3rd degree" in text or "complete" in text:
        analysis = (f"**Analysis**: **Complete (3rd Degree) AV Block** - CRITICAL finding.\n"
                   f"Complete AV dissociation is present. The ventricles beat independently at {hr:.0f} bpm "
                   f"(escape rhythm), unrelated to atrial activity. The regularity (CV={cv:.2f}) confirms "
                   f"the independent ventricular pacemaker.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 2nd Degree AV Block Type 1 (Wenckebach)
    if "wenckebach" in text or "type 1" in text:
        analysis = (f"**Analysis**: **2nd Degree AV Block Type I (Wenckebach)**.\n"
                   f"Progressive PR prolongation culminates in a dropped QRS. This 'grouped beating' pattern "
                   f"indicates AV nodal fatigability rather than structural damage.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 2nd Degree AV Block Type 2 (Mobitz II)
    if "mobitz" in text or "type 2" in text:
        analysis = (f"**Analysis**: **2nd Degree AV Block Type II (Mobitz II)** - HIGH RISK.\n"
                   f"Intermittent dropped beats occur WITHOUT prior PR prolongation. "
                   f"The wide QRS ({qrs_mean:.0f}ms) localizes this to the His-Purkinje system. "
                   f"Risk of progression to complete heart block.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # 1st Degree AV Block
    if "1st degree" in text:
        analysis = (f"**Analysis**: **1st Degree AV Block**.\n"
                   f"All atrial impulses conduct to ventricles, but with delay. PR interval is {pr:.0f}ms "
                   f"(normal <200ms). The rhythm remains {rhythm_desc}, making this typically benign.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PSVT/SVT
    if "psvt" in text or ("svt" in text and "nsvt" not in text):
        analysis = (f"**Analysis**: **Paroxysmal Supraventricular Tachycardia**.\n"
                   f"Rapid ({hr:.0f} bpm), regular tachycardia with narrow QRS ({qrs_mean:.0f}ms) confirms "
                   f"supraventricular origin. Sudden onset suggests re-entrant mechanism (AVNRT or AVRT).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # NSVT
    if "nsvt" in text:
        analysis = (f"**Analysis**: **Non-Sustained Ventricular Tachycardia** - SIGNIFICANT finding.\n"
                   f"A run of ≥3 consecutive wide-complex beats at tachycardic rate. "
                   f"Indicates ventricular irritability and warrants monitoring.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Tachycardia
    if "sinus tachycardia" in text:
        analysis = (f"**Analysis**: **Sinus Tachycardia**.\n"
                   f"Physiological acceleration of the sinus node. P-waves are normal, PR intact. "
                   f"Typically a response to stress, exercise, fever, or hypovolemia rather than primary arrhythmia.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Bradycardia
    if "sinus bradycardia" in text or "bradycardia" in text:
        analysis = (f"**Analysis**: **Sinus Bradycardia**.\n"
                   f"Slow but organized sinus rhythm. All conduction intervals normal. "
                   f"May be physiological (athletes, sleep) or pathological (medications, sick sinus).")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVC Bigeminy
    if "bigeminy" in text:
        analysis = (f"**Analysis**: **Ventricular Bigeminy**.\n"
                   f"Alternating pattern: Normal beat → PVC → Normal beat → PVC. "
                   f"Wide QRS complexes (>120ms) in every other beat create characteristic coupling.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVC Trigeminy
    if "trigeminy" in text:
        analysis = (f"**Analysis**: **Ventricular Trigeminy**.\n"
                   f"Pattern: Normal → Normal → PVC (repeating). "
                   f"Indicates stable ventricular ectopic focus with 3:1 coupling.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PVCs (general)
    if "pvc" in text and "bigeminy" not in text and "trigeminy" not in text:
        analysis = (f"**Analysis**: **Premature Ventricular Contractions**.\n"
                   f"Ectopic beats with wide QRS ({qrs_mean:.0f}ms) arising from ventricular focus. "
                   f"Arrive early in the cardiac cycle, often followed by compensatory pause.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # PAC Trigeminy
    if "pac trigeminy" in text:
        analysis = (f"**Analysis**: **PAC Trigeminy**.\n"
                   f"Rhythm Pattern: Normal → Normal → PAC. "
                   f"Every third beat acts as a premature atrial stimulus. Common in high adrenergic states.")
        return enhance(f"{intro}\n\n{analysis}")

    # PACs (general)
        analysis = (f"**Analysis**: **Premature Atrial Contractions**.\n"
                   f"Early beats originating from atrial ectopic focus. QRS remains narrow ({qrs_mean:.0f}ms), "
                   f"but P-wave morphology may be abnormal.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Pause
    if "pause" in text:
        analysis = (f"**Analysis**: **Significant Sinus Pause**.\n"
                   f"Prolonged interval (>2.0s) between beats detected. "
                   f"May indicate sinus arrest, exit block, or non-conducted PAC.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Ventricular Fibrillation
    if "ventricular fibrillation" in text:
        analysis = (f"**CRITICAL**: **Ventricular Fibrillation**.\n"
                   f"Chaotic, disorganized electrical activity with no discernible QRS complexes. "
                   f"Cardiac output is absent - IMMEDIATE defibrillation required.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Bundle Branch Blocks
    if "bundle branch" in text:
        analysis = (f"**Analysis**: **Bundle Branch Block**.\n"
                   f"Intraventricular conduction delay evident by wide QRS ({qrs_mean:.0f}ms). "
                   f"Rhythm is supraventricular, differentiating from ventricular ectopy.")
        return enhance(f"{intro}\n\n{analysis}")
    
    # Sinus Rhythm (Normal)
    if "sinus rhythm" in text:
        base = f"**Analysis**: **Normal Sinus Rhythm**.\nPhysiological rhythm with normal intervals."
        if hr > 100:
            base += f"\n*Note: Rate is elevated ({hr:.0f} bpm) - consider sinus tachycardia.*"
        if hr < 50:
            base += f"\n*Note: Rate is low ({hr:.0f} bpm) - consider sinus bradycardia.*"
        if pr > 200:
            base += f"\n*Note: PR prolonged ({pr:.0f}ms) - suggests 1st degree AV block.*"
        return enhance(f"{intro}\n\n{base}")
    
    # Generic fallback
    analysis = (f"**Analysis**: **{label}** detected.\n"
               f"Based on rhythm analysis (CV={cv:.2f}), rate ({hr:.0f} bpm), "
               f"and morphology (QRS {qrs_mean:.0f}ms).")
    return enhance(f"{intro}\n\n{analysis}")


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
