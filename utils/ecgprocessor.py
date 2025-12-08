import json
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
import psycopg2
from typing import List, Dict, Any, Tuple
import warnings
from pathlib import Path
import sys

# Suppress harmless scipy warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration Constants ---
TARGET_FS = 250  # Target sampling rate in Hz
SEGMENT_DURATION_S = 5.0  # Length of each segment in seconds
SEGMENT_LENGTH = int(TARGET_FS * SEGMENT_DURATION_S)
HRV_INTERP_FS = 4.0 # Resampling rate for RR intervals for PSD calculation (standard is 4 Hz)
# Define the root directory where your JSON files are located
DATA_ROOT_DIR = "." 

PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",  # <--- CHANGE THIS
    "password": "sais", # <--- CHANGE THIS
    "host": "127.0.0.1",
    "port": "5432"
}

class ECGProcessor:
    """
    Handles data loading, pre-processing, feature extraction (Time, Frequency, Non-Linear), 
    and database integration for heterogeneous ECG data.
    """

    def __init__(self, target_fs: int = TARGET_FS):
        self.target_fs = target_fs

    def _load_data_from_json(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Loads ECG data from the JSON file, accommodating heterogeneous structures."""
        file_path_obj = Path(file_path)
        filename = file_path_obj.name
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        signal = None
        original_fs = 250 # Default

        # Structure 1 & 3: Nested under "SensorData" (PTB-XL/Wearable)
        if isinstance(data.get("SensorData"), list) and data["SensorData"]:
            signal_data = data["SensorData"][0]
            if "ECG_CH_A" in signal_data:
                signal = np.array(signal_data["ECG_CH_A"], dtype=float)
            
            if "PTB-XL" in file_path or "00001_hr.json" in filename:
                original_fs = 360
            elif "ecg_segment_014.json" in filename:
                original_fs = 1000

        # Structure 2: Directly under the root (MIT-BIH style)
        elif "ECG_CH_A" in data:
            signal = np.array(data["ECG_CH_A"], dtype=float)
            if "MIT-BIH" in file_path or "104.json" in filename:
                 original_fs = 360 
        
        # Fallback for other files
        elif 'ECG_CH_B' in data: 
             signal = np.array(data["ECG_CH_B"], dtype=float)

        if signal is None:
            raise ValueError(f"Could not find valid ECG data channel in {filename}")

        # Basic Normalization
        if np.max(np.abs(signal)) > 5.0 and original_fs > 500:
            print(f"Normalizing raw signal data for {filename}.")
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        return signal, original_fs


    def _preprocess(self, signal: np.ndarray, original_fs: int) -> np.ndarray:
        """Resamples, filters, and removes baseline wander."""
        # 1. Resampling
        if original_fs != self.target_fs:
            resampled_signal = resample_poly(signal, self.target_fs, original_fs).astype(np.float32)
        else:
            resampled_signal = signal.astype(np.float32)

        nyquist = 0.5 * self.target_fs
        
        # 2. Baseline Wander Removal (0.5 Hz High-pass)
        b_hp, a_hp = butter(3, 0.5 / nyquist, btype='high', analog=False)
        baseline_removed = filtfilt(b_hp, a_hp, resampled_signal)

        # 3. High-Frequency Noise Reduction (40 Hz Low-pass)
        b_lp, a_lp = butter(3, 40.0 / nyquist, btype='low', analog=False)
        filtered_signal = filtfilt(b_lp, a_lp, baseline_removed)

        # 4. Powerline Noise Reduction (Notch filter at 50 & 60 Hz)
        # 50 Hz filter
        b_50, a_50 = butter(2, [ (50-1)/nyquist, (50+1)/nyquist], btype='bandstop', analog=False)
        filtered_signal = filtfilt(b_50, a_50, filtered_signal)
        # 60 Hz filter
        b_60, a_60 = butter(2, [ (60-1)/nyquist, (60+1)/nyquist], btype='bandstop', analog=False)
        final_signal = filtfilt(b_60, a_60, filtered_signal)
        
        return final_signal

    def _r_peak_detection(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Performs robust R-peak detection."""
        # Pan-Tompkins style steps (Differentiate, Square, Integrate)
        diff_signal = np.diff(signal)
        squared_signal = diff_signal**2
        window_size = int(0.150 * fs)
        window = np.ones(window_size) / window_size
        integrated_signal = np.convolve(squared_signal, window, mode='same')

        # Find peaks
        min_peak_distance = int(0.20 * fs)
        r_peaks, _ = find_peaks(integrated_signal, distance=min_peak_distance, height=np.mean(integrated_signal) * 0.7)

        # Refine peaks
        refined_r_peaks = []
        for peak in r_peaks:
            search_window = int(0.05 * fs)
            start = max(0, peak - search_window)
            end = min(len(signal), peak + search_window)
            max_idx = np.argmax(signal[start:end])
            refined_r_peaks.append(start + max_idx)

        return np.array(refined_r_peaks).astype(int)

    def _calculate_frequency_hrv(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """Calculates Frequency-Domain HRV features using Welch's method."""
        hrv_freq = {'VLF': 0.0, 'LF': 0.0, 'HF': 0.0, 'LF_HF_ratio': 0.0}
        
        if len(rr_intervals_ms) < 5: # Need at least 5 beats for stable spectrum
            return hrv_freq

        # 1. Create the time axis for the RR series (cumulative sum of RR times)
        rr_intervals_s = rr_intervals_ms / 1000
        rr_time_s = np.cumsum(rr_intervals_s)
        rr_time_s -= rr_time_s[0] 

        # 2. Resample the irregular RR series (RR_ms) onto a fixed, high-frequency grid (4 Hz)
        try:
            f_interp = interp1d(rr_time_s, rr_intervals_ms, kind='cubic')
            time_interp = np.arange(rr_time_s[0], rr_time_s[-1], 1/HRV_INTERP_FS)
            rr_interp = f_interp(time_interp)
        except ValueError:
            # Catch cases where rr_time_s is too short for interpolation
            return hrv_freq

        # 3. Calculate Power Spectral Density (PSD) using Welch's method
        fxx, pxx = welch(rr_interp, fs=HRV_INTERP_FS, window='hann', nperseg=256, noverlap=128)
        
        # 4. Integrate Power in Standard Bands
        vlf_band = (0.0, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        def band_power(f, p, band):
            idx = np.logical_and(f >= band[0], f < band[1])
            return np.trapz(p[idx], f[idx])

        vlf_power = band_power(fxx, pxx, vlf_band)
        lf_power = band_power(fxx, pxx, lf_band)
        hf_power = band_power(fxx, pxx, hf_band)

        hrv_freq['VLF'] = vlf_power
        hrv_freq['LF'] = lf_power
        hrv_freq['HF'] = hf_power
        
        if lf_power > 0 and hf_power > 0:
            hrv_freq['LF_HF_ratio'] = lf_power / hf_power
        else:
            hrv_freq['LF_HF_ratio'] = 0.0

        return hrv_freq
        
    def _calculate_nonlinear_hrv(self, rr_intervals_ms: np.ndarray) -> Dict[str, float]:
        """Calculates Non-Linear HRV features (Poincaré and Sample Entropy)."""
        hrv_nl = {'SD1': 0.0, 'SD2': 0.0, 'SampleEn': 0.0}
        
        if len(rr_intervals_ms) < 20: 
            return hrv_nl

        # 1. Poincaré Plot Parameters (SD1, SD2)
        rr_n = rr_intervals_ms[:-1]
        rr_n_plus_1 = rr_intervals_ms[1:]
        
        sd1 = np.sqrt(0.5 * np.var(rr_n - rr_n_plus_1))
        sd2 = np.sqrt(2 * np.var(rr_intervals_ms) - sd1**2)
        
        hrv_nl['SD1'] = sd1
        hrv_nl['SD2'] = sd2
        
        # 2. Sample Entropy 
        try:
            from biosppy.signals.hrv import sample_entropy
            # Using standard params: m=2, r=0.2 * std(RR)
            r = 0.2 * np.std(rr_intervals_ms)
            hrv_nl['SampleEn'] = sample_entropy(rr_intervals_ms, m=2, r=r)[0]
        except ImportError:
            hrv_nl['SampleEn'] = -1.0 
        except Exception:
            # Catch errors when sample_entropy fails (e.g., highly uniform RR intervals)
            hrv_nl['SampleEn'] = 0.0

        return hrv_nl

    def _calculate_morphology_features(self, segment: np.ndarray, segment_r_peaks: np.ndarray) -> Dict[str, float]:
        """Calculates QRS-focused morphology features."""
        morph_features = {'QRS_Avg_Energy': 0.0, 'QRS_Energy_Std': 0.0}
        
        if len(segment_r_peaks) == 0:
            return morph_features

        window_ms = 100 # +/- 100 ms
        window_samples = int(window_ms * self.target_fs / 1000)
        
        peak_energies = []
        
        for r_peak in segment_r_peaks:
            start = max(0, r_peak - window_samples)
            end = min(len(segment), r_peak + window_samples)
            
            qrs_segment = segment[start:end]
            energy = np.sum(qrs_segment**2)
            peak_energies.append(energy)

        if peak_energies:
            morph_features['QRS_Avg_Energy'] = np.mean(peak_energies)
            morph_features['QRS_Energy_Std'] = np.std(peak_energies)

        return morph_features

    def _sanitize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replaces all NaN and Inf values in the feature dictionary with 0.0 
        to ensure JSON compatibility for PostgreSQL JSONB type.
        """
        sanitized_features = {}
        for key, value in features.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    sanitized_features[key] = 0.0
                else:
                    sanitized_features[key] = value
            elif isinstance(value, np.floating):
                # Handle numpy float types, converting to standard float after check
                if np.isnan(value) or np.isinf(value):
                    sanitized_features[key] = 0.0
                else:
                    sanitized_features[key] = float(value)
            else:
                sanitized_features[key] = value
        return sanitized_features


    def _extract_segment_features(self, segment: np.ndarray, segment_r_peaks: np.ndarray, segment_idx: int) -> Dict[str, Any]:
        """
        Extracts all Time-Domain, Frequency-Domain, Non-Linear, and Morphology 
        features from a 5-second segment.
        """
        features = {}

        # 1. ECG Signal Shape Features (Baseline)
        features['mean_amplitude'] = np.mean(segment)
        features['std_amplitude'] = np.std(segment)
        features['max_amplitude'] = np.max(segment)
        features['min_amplitude'] = np.min(segment)
        features['signal_range'] = features['max_amplitude'] - features['min_amplitude']
        features['segment_index'] = segment_idx
        
        rr_intervals_ms = np.array([])
        if len(segment_r_peaks) >= 2:
            rr_intervals_samples = np.diff(segment_r_peaks)
            rr_intervals_ms = rr_intervals_samples * 1000 / self.target_fs

        # 2. Time-Domain HRV Features
        if len(rr_intervals_ms) > 0:
            features['mean_rr'] = np.mean(rr_intervals_ms)
            # Handle possible division by zero if mean_rr is 0 (shouldn't happen with R-peaks > 0)
            features['mean_hr'] = 60 / (features['mean_rr'] / 1000) if features['mean_rr'] > 0 else 0.0
            features['std_rr'] = np.std(rr_intervals_ms)
            
            features['SDNN'] = features['std_rr'] 
            features['RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals_ms)**2))
            
            nn_diff = np.abs(np.diff(rr_intervals_ms))
            features['pNN50'] = np.sum(nn_diff > 50) / len(nn_diff) if len(nn_diff) > 0 else 0
        else:
            default_hrv = {'mean_rr': 0.0, 'mean_hr': 0.0, 'std_rr': 0.0, 'SDNN': 0.0, 'RMSSD': 0.0, 'pNN50': 0.0}
            features.update(default_hrv)

        # 3. Morphology Features (QRS Energy)
        morph_feats = self._calculate_morphology_features(segment, segment_r_peaks)
        features.update(morph_feats)

        # 4. Frequency-Domain HRV Features
        freq_feats = self._calculate_frequency_hrv(rr_intervals_ms)
        features.update(freq_feats)

        # 5. Non-Linear HRV Features
        nl_feats = self._calculate_nonlinear_hrv(rr_intervals_ms)
        features.update(nl_feats)
            
        return features

    # --- Database Interaction Functions ---
    def _connect_to_db(self):
        """Establishes connection to the PostgreSQL database."""
        return psycopg2.connect(**PSQL_CONN_PARAMS)

    def _setup_database(self, conn):
        """Ensures the required table for feature storage exists."""
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS ecg_features_annotatable (
                    segment_id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    segment_index INT NOT NULL,
                    segment_start_s FLOAT NOT NULL,
                    segment_duration_s FLOAT NOT NULL,
                    arrhythmia_label VARCHAR(50) DEFAULT NULL,
                    r_peaks_in_segment TEXT, -- Segment-relative R-peak indices
                    features_json JSONB
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_segment ON ecg_features_annotatable (filename, segment_index);
            """)
            conn.commit()

    def process_and_save_record(self, file_path: str):
        """Main function to load, process, extract features, and save to database."""
        file_path_obj = Path(file_path)
        filename = file_path_obj.name
        print(f"\n--- Processing Record: {filename} ({file_path_obj.parent.name}) ---")
        
        try:
            raw_signal, original_fs = self._load_data_from_json(file_path)
            processed_signal = self._preprocess(raw_signal, original_fs)
            r_peaks_all = self._r_peak_detection(processed_signal, self.target_fs)
        except ValueError as e:
            print(f"Skipping {filename}: Data format error ({e})")
            return
        except Exception as e:
            print(f"Skipping {filename}: Pre-processing failed ({e})")
            return

        # Connect to DB
        try:
            conn = self._connect_to_db()
            self._setup_database(conn)
        except Exception as e:
            print(f"Database operation failed: {e}. Cannot save features. Check PSQL_CONN_PARAMS.")
            return

        # Segment and Extract Features
        num_segments = len(processed_signal) // SEGMENT_LENGTH
        print(f"Detected {len(processed_signal)} samples. Creating {num_segments} segments.")
        
        with conn.cursor() as cur:
            segments_processed = 0
            for i in range(num_segments):
                start_sample = i * SEGMENT_LENGTH
                end_sample = (i + 1) * SEGMENT_LENGTH
                segment = processed_signal[start_sample:end_sample]
                
                segment_r_peaks_indices = r_peaks_all[
                    (r_peaks_all >= start_sample) & (r_peaks_all < end_sample)
                ]
                segment_r_peaks_relative = segment_r_peaks_indices - start_sample

                features = self._extract_segment_features(segment, segment_r_peaks_relative, i)
                
                # --- FIX APPLIED HERE ---
                sanitized_features = self._sanitize_features(features)
                
                r_peaks_str = ",".join(map(str, segment_r_peaks_relative))
                segment_start_s = start_sample / self.target_fs
                
                try:
                    cur.execute("""
                        INSERT INTO ecg_features_annotatable 
                        (filename, segment_index, segment_start_s, segment_duration_s, r_peaks_in_segment, features_json)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (filename, segment_index) DO NOTHING
                    """, (
                        filename,
                        i,
                        segment_start_s,
                        SEGMENT_DURATION_S,
                        r_peaks_str,
                        json.dumps(sanitized_features) # Using sanitized features
                    ))
                    segments_processed += 1
                except psycopg2.Error as db_err:
                     # Log the specific segment failure but continue processing other segments
                    print(f"Database error on segment {i} of {filename}: {db_err}")
                
            conn.commit()
            print(f"Successfully inserted/updated {segments_processed} segments from {filename}.")
        
        conn.close()


# --- Execution Block ---
if __name__ == "__main__":
    
    # Check for biosppy for robust Sample Entropy calculation
    try:
        import biosppy
    except ImportError:
        print("\n*** WARNING: 'biosppy' not found. Install 'pip install biosppy' for robust Sample Entropy calculation. Defaulting to -1.0. ***\n")


    print("--- Starting Professional ECG Feature Extraction Pipeline ---")
    print(f"Searching for JSON files recursively in: {DATA_ROOT_DIR}")
    
    data_files = list(Path(DATA_ROOT_DIR).rglob('*.json'))
    
    if not data_files:
        print("ERROR: No JSON files found. Check your DATA_ROOT_DIR configuration.")
        sys.exit(1)
        
    print(f"Found {len(data_files)} JSON files to process.")
    print("--------------------------------------------------")

    processor = ECGProcessor()
    
    for file_path in data_files:
        try:
            processor.process_and_save_record(str(file_path))
        except psycopg2.Error as e:
            print(f"FATAL DB ERROR: Could not process {file_path}. Check database connection/permissions. Error: {e}")
            break
        except Exception as e:
            print(f"Critical failure during processing of {file_path}: {e}")
            
    print("\n--- Phase-1: ECG Processor Complete. ---")
    print("All segments and expanded features saved to 'ecg_features_annotatable' table.")