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

def _load_model():
    """
    Lazily load the CNN+Transformer model from checkpoints.
    Uses outputs/checkpoints/best_model.pth produced by train.py
    """
    global _model

    if _model is not None:
        return _model

    device = _init_device()

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"❌ No checkpoint found at {CKPT_PATH}. "
            "Train the model first (train.py) or ensure best_model.pth exists."
        )

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
            print(f"⚠️  Model checkpoint has different architecture: {e}")
            print(f"⚠️  Current CLASS_NAMES has {len(CLASS_NAMES)} classes")
            print(f"⚠️  Please retrain the model or use the old class list")
            raise RuntimeError(
                f"Model checkpoint incompatible with current {len(CLASS_NAMES)} classes. "
                "The checkpoint was trained with a different number of classes. "
                "Please retrain the model with the new class list."
            )
        
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
        print(f"✓ Model loaded from {CKPT_PATH}")
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
            if 40 < hr < 100:
                # Normal sinus rhythm
                return ("Sinus Rhythm", "HIGH",
                        f"Regular rhythm (CV={cv:.3f}) with normal HR={hr:.0f} bpm - NOT AFib")
            elif hr >= 100:
                # Sinus tachycardia
                if 120 <= pr <= 200 and qrs_mean < 120:
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
    # Criteria: VERY irregular RR intervals (high CV and RMSSD)
    # Made STRICTER to avoid false positives on regular rhythms
    if len(rr_arr) > 3:
        rr_std = np.std(rr_arr)
        rr_mean = np.mean(rr_arr)
        cv = rr_std / rr_mean if rr_mean > 0 else 0
        
        # AFib: VERY irregular rhythm (CV > 0.20) AND high RMSSD
        # Increased threshold from 0.15 to 0.20 to be more conservative
        if cv > 0.20 and rmssd > 60:
            return ("Atrial Fibrillation", "HIGH", 
                    f"Highly irregular RR intervals (CV={cv:.2f}) with RMSSD={rmssd:.0f}ms")
        # Moderate AFib detection
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
    # RULE 4: 1st Degree AV Block
    # ============================================================
    # Criteria: PR interval > 200 ms (or > 180ms for moderate)
    if pr > 200:
        return ("1st Degree AV Block", "HIGH",
                f"Prolonged PR interval ({pr:.0f}ms > 200ms)")
    elif pr > 180 and pr <= 200:
        # Borderline prolonged PR
        return ("1st Degree AV Block", "MODERATE",
                f"Borderline prolonged PR interval ({pr:.0f}ms)")
    
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
        if len(qrs_list) >= 3:
            for i in range(len(qrs_list) - 2):
                if qrs_list[i] > 120 and qrs_list[i+1] > 120 and qrs_list[i+2] > 120:
                    return ("PVC Triplets (VT Warning)", "HIGH",
                           f"Three consecutive PVCs! Warning for VT at position {i}-{i+2}")
        
        # PVC Couplets: 2 consecutive PVCs (SECOND PRIORITY)
        if len(qrs_list) >= 2:
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
                if cv < 0.1:
                    return ("Supraventricular Tachycardia", "HIGH",
                            f"Very high HR ({hr:.0f} bpm) with narrow QRS and regular rhythm")
    
    # ============================================================
    # No strong rule match
    # ============================================================
    return (None, None, None)


def _clinical_explanation(label: str, features: dict) -> str:
    """
    Returns a text explanation combining:
      - the predicted class
      - key features (HR, PR interval, QRS durations)
    CLASS_NAMES (current):
      0: Sinus Rhythm
      1: Supraventricular Tachycardia
      2: Atrial Fibrillation
      3: PVCs
    """

    if not label:
        return "No arrhythmia detected in this segment."

    text = label.lower()
    
    # Safely get HR and PR, handling None
    hr_val = features.get("mean_hr")
    hr = float(hr_val) if hr_val is not None else 0.0
    
    pr_val = features.get("pr_interval")
    pr = float(pr_val) if pr_val is not None else 0.0

    # Safely calculate QRS mean, filtering out None values
    qrs_mean = 0.0
    try:
        raw_qrs = features.get("qrs_durations_ms")
        if isinstance(raw_qrs, list):
            qrs_list = [x for x in raw_qrs if x is not None and isinstance(x, (int, float))]
        elif isinstance(raw_qrs, str):
            # Handle case where it's stored as a string
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

    # Helper to format metrics nicely
    def fmt_vitals():
        parts = []
        if hr > 10: parts.append(f"HR ≈ {hr:.0f} bpm")
        if pr > 10: parts.append(f"PR ≈ {pr:.0f} ms")
        if qrs_mean > 10: parts.append(f"QRS ≈ {qrs_mean:.0f} ms")
        return ", ".join(parts)

    v_str = fmt_vitals()
    suffix = f" ({v_str})" if v_str else ""

    # Atrial Fibrillation
    if "atrial fibrillation" in text or "afib" in text:
        return (
            "Atrial Fibrillation is suspected. The model detects an irregular rhythm "
            "with absence of consistent P-waves and variable RR intervals." + suffix
        )

    # Supraventricular Tachycardia
    if "supraventricular tachycardia" in text or "svt" in text:
        return (
            "Supraventricular Tachycardia pattern: rapid, regular rhythm originating above the ventricles. "
            f"Likely narrow QRS complexes.{suffix}"
        )

    # --- ECTOPIC BEATS & PATTERNS ---
    
    if "bigeminy" in text:
        return (
            f"Bigeminy detected: Every other beat is an ectopic beat.{suffix} "
            "This indicates a high burden of prematurity (50% of beats). "
            "If Ventricular: Risk of cardiomyopathy if untreated. "
            "If Atrial: Often benign but can trigger AFib."
        )
    
    if "trigeminy" in text:
        return (
            f"Trigeminy detected: Every third beat is an ectopic beat (N-N-E pattern).{suffix}"
        )
    
    if "quadrigeminy" in text:
        return (
            f"Quadrigeminy detected: Every fourth beat is an ectopic beat.{suffix}"
        )
    
    if "couplet" in text or "pair" in text:
        return (
            f"Couplet/Pair detected: Two consecutive ectopic beats.{suffix} "
            "This indicates increased electrical instability."
        )
    
    if "triplet" in text or "vt warning" in text or "run" in text:
        return (
            f"⚠️ Run of Ectopy / Triplet detected: 3+ consecutive ectopic beats.{suffix} "
            "If Ventricular (NSVT): Significant risk marker. Requires cardiac evaluation."
        )
    
    # PACs (Single)
    if "pac" in text and not ("pair" in text or "run" in text):
         return (
             "Premature Atrial Contractions (PACs) suspected. "
             f"Early beats with abnormal P-wave morphology, followed by normal QRS.{suffix}"
         )

    # General PVCs
    if "pvc" in text and not ("bigeminy" in text or "trigeminy" in text):
        base = (
            "Premature Ventricular Contractions (PVCs) suspected. "
            "Wide QRS complexes without preceding P-waves."
        )
        if v_str:
            base += f" ({v_str})"
        return base

    # --- ADVANCED RHYTHMS ---

    if "junctional" in text:
        hr_status = "Bradycardia (Escape rhythm)." if hr > 0 and hr < 60 else "Accelerated or Tachycardia."
        return (
            f"Junctional Rhythm: Originates from the AV node. "
            f"Usually narrow QRS with absent or inverted P-waves. {hr_status}{suffix}" 
        )

    if "idioventricular" in text:
        return (
            "Idioventricular Rhythm (IVR): Ventricular escape rhythm. "
            "Wide QRS complexes, very slow rate (< 40 bpm), independent of atria. "
            "Last resort pacemaker of the heart. EMERGENCY if patient is unstable."
        )
    
    if "ventricular fibrillation" in text:
        return (
            "⚠️ VENTRICULAR FIBRILLATION (VFib): "
            "Chaotic ventricular activity. No effective cardiac output. "
            "CARDIAC ARREST - REQUIRES IMMEDIATE DEFIBRILLATION."
        )

    if "atrial flutter" in text:
        return (
            "Atrial Flutter: Macro-reentrant atrial tachycardia. "
            "Characteristic 'Sawtooth' F-waves @ ~300bpm. "
            f"Ventricular rate often fixed ratio e.g., 2:1 block.{suffix}"
        )

    if "atrial fibrillation" in text:
        return (
            "Atrial Fibrillation: Irregularly irregular rhythm with no distinct P-waves. "
            f"Risk of stroke due to clot formation.{suffix}"
        )

    # Blocks
    if "wenckebach" in text or "type 1" in text:
        return (
            "2nd Degree AV Block Type 1 (Wenckebach): "
            "Progressive prolongation of PR interval until a beat is dropped. "
            f"Usually benign.{suffix}"
        )
        
    if "mobitz ii" in text or "type 2" in text:
        return (
            "⚠️ 2nd Degree AV Block Type 2 (Mobitz II): "
            "Intermittent dropped beats with constant PR interval. "
            f"High risk of progression to Complete Heart Block. Pacemaker indicated.{suffix}"
        )

    if "3' av block" in text or "complete" in text:
        return (
            "⚠️ 3rd Degree (Complete) AV Block: "
            "Total dissociation between atria (P-waves) and ventricles (QRS). "
            f"Ventricular rate is slow ({hr:.1f} bpm) and regular. Pacemaker required."
        )
    
    if "bundle branch" in text:
        return (
            "Bundle Branch Block (BBB): Conduction delay in His-Purkinje system. "
            "Wide QRS complex (> 120ms). "
            "RBBB has RSR' in V1. LBBB has broad notched R in V6."
        )

    # Sinus Tachycardia (Explicit Class)
    if "sinus tachycardia" in text:
         return (
            f"Sinus Tachycardia: The rhythm is sinus but fast (HR > 100 bpm). "
            f"Detected HR ≈ {hr:.1f} bpm. PR interval ≈ {pr:.0f} ms."
         )

    # Sinus Bradycardia (Explicit Class)
    if "sinus bradycardia" in text:
         return (
            f"Sinus Bradycardia: The rhythm is sinus but slow (HR < 60 bpm). "
            f"Detected HR ≈ {hr:.1f} bpm. PR interval ≈ {pr:.0f} ms."
         )
    
    # 1st Degree AV Block
    if "av block" in text:
         return (
             f"1st Degree AV Block: Prolonged PR interval (> 200ms). "
             f"Detected PR ≈ {pr:.0f} ms. HR ≈ {hr:.1f} bpm."
         )

    # Sinus Rhythm (Normal) - check for hidden conditions
    if "sinus rhythm" in text:
        # Check for contradictions or hidden arrhythmias based on rules
        if hr > 100:
             return (
                f"Model predicts Sinus Rhythm, but Heart Rate is elevated ({hr:.1f} bpm). "
                "This suggests Sinus Tachycardia."
             )
        if hr < 50:
             return (
                f"Model predicts Sinus Rhythm, but Heart Rate is low ({hr:.1f} bpm). "
                "This suggests Sinus Bradycardia."
             )
        if pr > 200:
             return (
                 f"Model predicts Sinus Rhythm, but PR interval is prolonged ({pr:.0f} ms). "
                 "This suggests 1st Degree AV Block."
             )
        
        return (
            f"Sinus Rhythm: Normal P-QRS-T pattern with regular rate. "
            f"HR ≈ {hr:.1f} bpm, PR ≈ {pr:.0f} ms."
        )

    return (
        f"Model predicts {label}. Mean HR ≈ {hr:.1f} bpm, PR ≈ {pr:.0f} ms, "
        f"QRS ≈ {qrs_mean:.0f} ms. No other clear pattern detected."
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
        override_reason = None
        
        if rule_prediction is not None:
            # Rule detected something
            if rule_prediction != model_prediction:
                # Conflict between model and rules
                # PRIORITY OVERRIDE for numerical facts (HR, PR)
                numerical_facts = ["Sinus Tachycardia", "Sinus Bradycardia", "1st Degree AV Block"]
                
                if rule_confidence == "HIGH" or (rule_prediction in numerical_facts):
                    # Strong clinical evidence OR numerical fact - override model
                    final_prediction = rule_prediction
                    override_reason = f"Clinical rules override: {rule_reason}"
                    print(f"⚠️  OVERRIDE: Model said '{model_prediction}', but rules detected '{rule_prediction}' ({rule_reason})")
                else:
                    # Moderate evidence - mention in explanation but keep model prediction
                    override_reason = f"Note: {rule_reason}, but model confidence is high"
            else:
                # Model and rules agree
                override_reason = f"Confirmed by clinical rules: {rule_reason}"
        
        pred_label = final_prediction
        # --- END HYBRID APPROACH ---

        saliency = _compute_saliency(model, x, pred_idx)
        explanation = _clinical_explanation(pred_label, features or {})
        
        # Don't add override info to explanation - it's already in console logs

        return {
            "pred_label": pred_label,
            "model_prediction": model_prediction,  # Also return what model said
            "probabilities": probs.tolist(),
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
