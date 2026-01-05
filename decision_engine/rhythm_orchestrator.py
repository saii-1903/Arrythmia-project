
import numpy as np

# This is a stub for rule-based logic to be imported from xai or similar
# For now, we define a simple interface or import the existing logic if possible.
# Ideally, xai.py's clinical rules should be separated, but to avoid huge refactors
# we will mimic the logic here or import XAI if it provides a clean rule function.

def apply_clinical_rules(features: dict):
    """
    Apply identifying rules (e.g. Rate checks, Regularity checks).
    Returns (rule_label, confidence, reason) or (None, 0.0, "")
    """
    # Extract
    hr = features.get("mean_hr", 0)
    rr_cv = features.get("rr_variability", 0)
    pr = features.get("pr_interval", 0)
    qrs_width = features.get("qrs_width", 0)
    
    # 1. Asystole / Pause Check (simplistic)
    # If not handled by artifact check, could handle here.
    
    # 2. Tachy/Brady
    if hr > 150:
        return "Supraventricular Tachycardia", 0.7, "HR > 150 bpm"
    if hr < 45:
        return "Sinus Bradycardia", 0.8, "HR < 45 bpm"
    
    # 3. AFIB Check (Irregularity)
    if rr_cv > 0.15: # High variability
         return "Atrial Fibrillation", 0.65, "High RR Irregularity (> 15%)"
         
    return None, 0.0, ""

class RhythmOrchestrator:
    def __init__(self):
        pass

    def decide(self, 
               ml_prediction: dict,   # {label, probs, confidence}
               clinical_features: dict, 
               sqi_result: dict) -> dict:
        """
        Arbitrate between Signal Quality, Rules, and ML.
        
        Priority:
        1. SQI Fail -> Artifact
        2. Strong Rule Match -> Rule Label
        3. ML Prediction
        """
        
        # 1. Gating
        if not sqi_result['is_acceptable']:
            return {
                "final_label": "Artifact / Noise",
                "confidence": 1.0,
                "source": "SQI_Gating",
                "explanation": f"Signal Rejected: {', '.join(sqi_result['issues'])}",
                "probabilities": [0.0]*len(ml_prediction.get('probs',[])) # Dummy
            }
            
        # 2. Rule Check
        rule_lbl, rule_conf, rule_reason = apply_clinical_rules(clinical_features)
        
        # 3. ML Output
        ml_lbl = ml_prediction.get("label", "Unknown")
        ml_conf = ml_prediction.get("confidence", 0.0)
        ml_probs = ml_prediction.get("probs", [])
        
        # 4. Arbitration Logic
        
        # Conflict: Rule says AFib, Model says Sinus?
        # If Rule Confidence is Very High (> 0.9), trust Rule?
        # For now, let's say Rule acts as a 'modifier' or 'override' for safety
        
        final_label = ml_lbl
        final_conf = ml_conf
        source = "ML_Model"
        final_reason = f"Model predicts {ml_lbl} with {ml_conf:.2f} confidence."
        
        # Safety Overrides
        # If ML says NORM but HR > 150 -> Trust Rule (SVT)
        if ml_lbl in ["Sinus Rhythm", "Sinus Tachycardia"] and rule_lbl == "Supraventricular Tachycardia":
            final_label = rule_lbl
            final_conf = rule_conf
            source = "Clinical_Rule_Override"
            final_reason = f"Model predicted {ml_lbl}, but HR > 150 indicates SVT."
            
        # If ML says NORM but Irregularity is HUGE -> Suggest AFib check
        if ml_lbl == "Sinus Rhythm" and rule_lbl == "Atrial Fibrillation":
            # Soft override or warning?
            # Let's override if ML confidence is low
            if ml_conf < 0.8:
                final_label = "Atrial Fibrillation"
                final_conf = rule_conf
                source = "Clinical_Rule_Weak_ML"
                final_reason = "Model unsure (conf < 0.8) and high irregularity suggests AFib."

        return {
            "final_label": final_label,
            "confidence": final_conf,
            "source": source,
            "explanation": final_reason,
            "probabilities": ml_probs
        }
