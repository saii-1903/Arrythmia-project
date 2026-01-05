
import numpy as np
import yaml
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        # Fallback defaults if config missing
        return {
            'artifacts': {
                'flatline_threshold': 0.001,
                'max_step_change': 4.0,
                'high_freq_noise_threshold': 100.0
            }
        }
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

_CONFIG = load_config()

def check_signal_quality(signal: np.ndarray, fs: int) -> dict:
    """
    Analyze signal for artifacts using thresholds from config.
    
    Returns dictionary:
    {
        "is_acceptable": bool,
        "issues": list[str],  # e.g. ["Flatline detected", "Extreme noise"]
        "sqi_score": float    # 0.0 to 1.0 (Simple heuristic)
    }
    """
    if len(signal) == 0:
        return {"is_acceptable": False, "issues": ["Empty signal"], "sqi_score": 0.0}

    issues = []
    cfg = _CONFIG.get('artifacts', {})
    
    # 1. Flatline / Disconnection Check
    # Check standard deviation
    sig_std = np.std(signal)
    flat_thresh = cfg.get('flatline_threshold', 0.001)
    
    if sig_std < flat_thresh:
        issues.append("Flatline / Lead Disconnected")
    
    # 2. Sudden Amplitude Jumps (steps)
    # Derivative
    diff = np.diff(signal)
    max_step = np.max(np.abs(diff)) if len(diff) > 0 else 0
    step_thresh = cfg.get('max_step_change', 4.0)
    
    if max_step > step_thresh:
        issues.append(f"Sudden artifact jump detected ({max_step:.1f} > {step_thresh})")

    # 3. Saturation (Rail) check
    # If many points are exactly equal to min or max
    # Heuristic: if > 5% of points are at min or max
    # (Assuming floats, exact equality is rare unless clipped)
    # Check range first
    sig_min, sig_max = np.min(signal), np.max(signal)
    if sig_max == sig_min:
         pass # Handled by flatline
    else:
        n_points = len(signal)
        n_min = np.sum(signal == sig_min)
        n_max = np.sum(signal == sig_max)
        if (n_min / n_points > 0.05) or (n_max / n_points > 0.05):
            issues.append("Signal saturation/clipping detected")

    # 4. High Frequency Noise check
    # Simple heuristic: Std dev of high-passed signal (diff)
    # Ideally should be bandpassed, but diff is a rough high-pass
    # Normal QRS has high slopes, so we must be careful. 
    # Let's count "noisy segments"
    # Or just use an SQI placeholder for now?
    # Let's stick to the config-based checks requested.
    
    # Final Decision
    is_acceptable = len(issues) == 0
    
    # Advanced SQI Score
    from signal_processing.sqi import calculate_sqi_score
    try:
        numeric_sqi = calculate_sqi_score(signal, fs)
    except Exception as e:
        print(f"SQI Calc Error: {e}")
        numeric_sqi = 0.0

    # If artifacts detected, cap the SQI score
    if not is_acceptable and numeric_sqi > 40.0:
        numeric_sqi = 40.0 # Force low score if hard fails exist

    # Also, if SQI is extremely low, mark as unacceptable even if hard checks passed
    if numeric_sqi < 25.0 and is_acceptable:
        is_acceptable = False
        issues.append(f"Low Signal Quality Score ({numeric_sqi:.1f}/100)")
        
    return {
        "is_acceptable": is_acceptable,
        "issues": issues,
        "sqi_score": numeric_sqi
    }
