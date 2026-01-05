
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def calculate_sqi_score(signal: np.ndarray, fs: int) -> float:
    """
    Calculate a quantitative Signal Quality Index (SQI) from 0.0 to 100.0.
    Based on:
      1. R-peak Consistency (RR interval variance) - implicitly handled by robust detectors usually, 
         but here we check basic signal stats. 
      2. SNR (Signal-to-Noise Ratio) estimate
      3. Baseline Stability (Zero-crossing rate approx)
      4. Physiologic range checks (Skewness/Kurtosis)
    """

    if len(signal) == 0:
        return 0.0

    score = 100.0
    deductions = 0.0

    # ----------------------------------------
    # 1. Amplitude / Flatline Check
    # ----------------------------------------
    amp_range = np.max(signal) - np.min(signal)
    if amp_range < 0.05:  # < 0.05 mV is basically noise/flatline
        return 0.0  # Immediate fail
    if amp_range > 10.0:  # > 10 mV is likely saturation/artifact
        deductions += 40

    # ----------------------------------------
    # 2. Skewness & Kurtosis (Morphology Check)
    # ----------------------------------------
    # Good ECGs are typically skewed (R-peaks) and have high kurtosis (peakedness).
    # Gaussian noise has skew ~ 0, kurtosis ~ 3.
    try:
        k = kurtosis(signal)  # Fisher kurtosis, normal = 0 (Pearson = 3)
        s = skew(signal)
        
        # Good ECG usually has Kurtosis > 5 (high peaks relative to baseline)
        if k < 5.0:
            deductions += 20
        # If kurtosis is extremely low (<1), it looks like noise
        if k < 1.0:
            deductions += 20

    except Exception:
        deductions += 10 # Calculation error

    # ----------------------------------------
    # 3. Power Spectrum (Baseline vs QRS)
    # ----------------------------------------
    # Check if dominant energy is in physiological range (1-40 Hz) vs noise
    try:
        f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), 1024))
        
        # Total power
        total_p = np.sum(Pxx)
        if total_p == 0: return 0.0

        # Power in 50/60Hz band (Mains Noise)
        idx_50 = np.logical_and(f >= 48, f <= 52)
        idx_60 = np.logical_and(f >= 58, f <= 62)
        power_noise = np.sum(Pxx[idx_50]) + np.sum(Pxx[idx_60])
        
        noise_ratio = power_noise / total_p
        if noise_ratio > 0.1: # > 10% energy is mains hum
            deductions += 30

        # Power in baseline (< 0.5 Hz)
        idx_base = f < 0.5
        power_base = np.sum(Pxx[idx_base])
        baseline_ratio = power_base / total_p
        
        if baseline_ratio > 0.4: # > 40% energy is drift
            deductions += 25
            
    except Exception:
        pass

    # ----------------------------------------
    # 4. Final Score
    # ----------------------------------------
    final_score = max(0.0, score - deductions)
    return float(final_score)
