
import yaml
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

_CONFIG = load_config()

def get_target_fs():
    return _CONFIG['sampling']['target_fs']

def remove_baseline_wander(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Remove baseline wander using High-Pass filter defined in config.
    Default: Butterworth HP @ 0.5 Hz
    """
    cfg = _CONFIG['baseline']
    if cfg['method'] == 'butterworth_highpass':
        cutoff = cfg['cutoff']
        order = cfg['order']
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal)
    
    # Fallback or other methods can be added here
    return signal

def remove_powerline_noise(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Remove 50Hz/60Hz noise using Notch filters defined in config.
    """
    cfg = _CONFIG['powerline']
    if cfg['method'] == 'notch':
        freqs = cfg['frequencies']
        q = cfg['quality_factor']
        
        cleaned = signal.copy()
        for f0 in freqs:
            # Skip if f0 is >= Nyquist
            if f0 >= 0.5 * fs:
                continue
                
            b, a = iirnotch(f0, q, fs)
            cleaned = filtfilt(b, a, cleaned)
        return cleaned
    
    return signal

def clean_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply full cleaning pipeline: Baseline Removal -> Powerline Removal.
    """
    # 1. Baseline
    sig_no_base = remove_baseline_wander(signal, fs)
    
    # 2. Powerline
    sig_clean = remove_powerline_noise(sig_no_base, fs)
    
    return sig_clean
