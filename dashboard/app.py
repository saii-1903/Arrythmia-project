from flask import Flask, render_template, jsonify, request, redirect, url_for
import sys
import os
from pathlib import Path

# --- FOLDER RESTRUCTURE FIX ---
# Add project root and sibling folders to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "database"))
sys.path.append(str(BASE_DIR / "xai"))
sys.path.append(str(BASE_DIR / "models_training"))

import db_service
import json
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename
from typing import List, Dict, Any, Tuple
import warnings
import psycopg2
import subprocess

# XAI – Option A (clinical text + model prediction)
from xai import explain_segment, reset_model
from data_loader import CLASS_NAMES

# Suppress harmless scipy warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# Flask App & Global Config
# =========================================================

app = Flask(__name__)

TARGET_FS = 250
SEGMENT_DURATION_S = 10.0
SEGMENT_LENGTH = int(TARGET_FS * SEGMENT_DURATION_S)
HRV_INTERP_FS = 4.0

# Where individual uploaded JSONs go
# Point to shared 'data/ecg_data' folder
DATA_ROOT = BASE_DIR / "data"
app.config["UPLOAD_FOLDER"] = str(DATA_ROOT / "ecg_data")
DATA_ROOT_DIR = Path(app.config["UPLOAD_FOLDER"])
os.makedirs(DATA_ROOT_DIR, exist_ok=True)

# Folder that holds the bulk JSON datasets already converted
DATASET_JSON_DIR = DATA_ROOT / "input_segments"
os.makedirs(DATASET_JSON_DIR, exist_ok=True)


# =========================================================
# ECG Loading & Preprocessing
# =========================================================

def _load_data_from_json(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads ECG data from a JSON file and returns (signal, original_fs).
    Supports:
      - SensorData[0]["ECG_CH_A"]
      - Top-level "ECG_CH_A" / "ECG_CH_B"
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    signal = None
    original_fs = 250
    filename = file_path.name

    # Structure 1: SensorData list (PTB-XL style, etc.)
    if isinstance(data.get("SensorData"), list) and data["SensorData"]:
        row = data["SensorData"][0]
        if "ECG_CH_A" in row:
            signal = np.array(row["ECG_CH_A"], dtype=float)

        # Heuristics for fs
        if "PTBXL" in filename.upper() or "PTB-XL" in filename.upper():
            original_fs = 500  # many PTB-XL records
        elif "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    # Structure 2: top-level "ECG_CH_A"
    elif "ECG_CH_A" in data:
        signal = np.array(data["ECG_CH_A"], dtype=float)
        if "MITDB" in filename.upper() or "MIT-BIH" in filename.upper():
            original_fs = 360

    elif "ECG_CH_B" in data:
        signal = np.array(data["ECG_CH_B"], dtype=float)

    if signal is None:
        raise ValueError(f"Could not find valid ECG channel in JSON file: {filename}")

    return signal, original_fs


def _preprocess(signal: np.ndarray, original_fs: int) -> np.ndarray:
    """
    Resample to TARGET_FS and apply:
      - High-pass (0.5 Hz)
      - Low-pass (40 Hz)
      - 50 Hz + 60 Hz notch
    """
    if original_fs != TARGET_FS:
        signal = resample_poly(signal, TARGET_FS, original_fs).astype(np.float32)
    else:
        signal = signal.astype(np.float32)

    nyq = 0.5 * TARGET_FS

    # High-pass for baseline wander
    b_hp, a_hp = butter(3, 0.5 / nyq, btype="high")
    signal = filtfilt(b_hp, a_hp, signal)

    # Low-pass for HF noise
    b_lp, a_lp = butter(3, 40.0 / nyq, btype="low")
    signal = filtfilt(b_lp, a_lp, signal)

    # 50 Hz notch
    b_50, a_50 = butter(2, [(50 - 1) / nyq, (50 + 1) / nyq], btype="bandstop")
    signal = filtfilt(b_50, a_50, signal)

    # 60 Hz notch
    b_60, a_60 = butter(2, [(60 - 1) / nyq, (60 + 1) / nyq], btype="bandstop")
    signal = filtfilt(b_60, a_60, signal)

    return signal


def _r_peak_detection(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Modified Pan–Tompkins using:
      - diff -> square -> moving integration
      - refine peak on raw channel
    """
    diff_signal = np.diff(signal)
    squared_signal = diff_signal**2

    # Integration window ~150 ms
    window_size = int(0.150 * fs)
    window = np.ones(window_size) / window_size
    integrated_signal = np.convolve(squared_signal, window, mode="same")

    min_peak_distance = int(0.20 * fs)
    r_peaks, _ = find_peaks(
        integrated_signal,
        distance=min_peak_distance,
        height=np.mean(integrated_signal) * 0.7,
    )

    refined = []
    for p in r_peaks:
        sw = int(0.05 * fs)
        start = max(0, p - sw)
        end = min(len(signal), p + sw)
        local = signal[start:end]
        if local.size == 0:
            continue
        max_idx = np.argmax(local)
        refined.append(start + max_idx)

    return np.array(refined, dtype=int)


# =========================================================
# HRV, Morphology & PR/QRS Features
# =========================================================

def _calculate_frequency_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Frequency-domain HRV (VLF, LF, HF, LF/HF) using Welch.
    """
    out = {"VLF": 0.0, "LF": 0.0, "HF": 0.0, "LF_HF_ratio": 0.0}
    if len(rr_intervals_ms) < 5:
        return out

    rr_s = rr_intervals_ms / 1000.0
    t = np.cumsum(rr_s)
    t -= t[0]

    try:
        f_interp = interp1d(t, rr_intervals_ms, kind="cubic")
        t_new = np.arange(t[0], t[-1], 1.0 / HRV_INTERP_FS)
        rr_interp = f_interp(t_new)
    except ValueError:
        return out

    n = len(rr_interp)
    if n < 16:
        return out

    nperseg = min(n, 256)
    noverlap = nperseg // 2

    fxx, pxx = welch(
        rr_interp,
        fs=HRV_INTERP_FS,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
    )

    def band_power(f, p, band):
        idx = (f >= band[0]) & (f < band[1])
        if not np.any(idx):
            return 0.0
        return float(np.trapz(p[idx], f[idx]))

    vlf = band_power(fxx, pxx, (0.0, 0.04))
    lf = band_power(fxx, pxx, (0.04, 0.15))
    hf = band_power(fxx, pxx, (0.15, 0.4))

    out["VLF"] = vlf
    out["LF"] = lf
    out["HF"] = hf
    out["LF_HF_ratio"] = float(lf / hf) if lf > 0 and hf > 0 else 0.0
    return out


def _calculate_nonlinear_hrv(rr_intervals_ms: np.ndarray) -> Dict[str, float]:
    """
    Poincaré SD1/SD2.
    """
    out = {"SD1": 0.0, "SD2": 0.0}
    if len(rr_intervals_ms) < 2:
        return out

    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]

    sd1 = np.sqrt(0.5 * np.var(rr_n - rr_n1))
    sd2 = np.sqrt(2 * np.var(rr_intervals_ms) - sd1**2)

    out["SD1"] = float(sd1)
    out["SD2"] = float(sd2)
    return out


def _compute_qrs_durations(segment: np.ndarray, segment_r_peaks: np.ndarray, fs: int) -> np.ndarray:
    """
    Estimate QRS duration:
      - 200 ms window around each R
      - threshold at 50% of local peak amplitude
      - find left/right crossings
    """
    if segment_r_peaks is None or len(segment_r_peaks) == 0:
        return np.array([])

    durations = []
    half_window = int(0.10 * fs)

    for r in segment_r_peaks:
        r = int(r)
        start = max(0, r - half_window)
        end = min(len(segment) - 1, r + half_window)
        local = segment[start : end + 1]
        if local.size == 0:
            continue

        peak_val = segment[r]
        thresh = 0.5 * peak_val

        # Left
        left_idx = r
        for i in range(r, start, -1):
            if (segment[i] - thresh) * (segment[i - 1] - thresh) <= 0:
                left_idx = i
                break

        # Right
        right_idx = r
        for i in range(r, end):
            if (segment[i] - thresh) * (segment[i + 1] - thresh) <= 0:
                right_idx = i
                break

        width_samples = max(1, right_idx - left_idx)
        width_ms = width_samples * 1000.0 / fs

        if 40 <= width_ms <= 200:
            durations.append(width_ms)

    return np.array(durations, dtype=float)


def _calculate_morphology_features(segment: np.ndarray, segment_r_peaks: np.ndarray) -> Dict[str, Any]:
    """
    QRS energy + QRS duration distribution (ms).
    """
    out: Dict[str, Any] = {
        "QRS_Avg_Energy": 0.0,
        "QRS_Energy_Std": 0.0,
        "qrs_durations_ms": [],
    }
    if len(segment_r_peaks) == 0:
        return out

    window_samples = int(0.100 * TARGET_FS)
    energies = []

    for r in segment_r_peaks:
        start = max(0, r - window_samples)
        end = min(len(segment), r + window_samples)
        qrs_seg = segment[start:end]
        energies.append(np.sum(qrs_seg**2))

    if energies:
        out["QRS_Avg_Energy"] = float(np.mean(energies))
        out["QRS_Energy_Std"] = float(np.std(energies))

    qrs_list = _compute_qrs_durations(segment, segment_r_peaks, TARGET_FS)
    out["qrs_durations_ms"] = qrs_list.tolist() if qrs_list.size > 0 else []
    return out


def _sanitize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure no NaN/Inf for JSONB."""
    clean = {}
    for k, v in features.items():
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def _calculate_pr_interval(signal: np.ndarray, r_peaks: np.ndarray, fs: int) -> float:
    """
    Estimate PR interval:
      - for each R, look back 250 ms
      - detect P peak
      - keep PR in [50, 300] ms
    """
    if r_peaks is None or len(r_peaks) == 0:
        return 0.0

    pr_vals = []
    lookback = int(0.25 * fs)

    for r in r_peaks:
        r = int(r)
        start = max(0, r - lookback)
        end = r
        if end - start < 5:
            continue

        pre_seg = signal[start:end]
        if pre_seg.size == 0:
            continue

        try:
            prom = max(0.01, np.std(pre_seg) * 0.15)
        except Exception:
            prom = 0.01

        p_peaks_rel, _ = find_peaks(
            pre_seg, prominence=prom, distance=int(0.04 * fs)
        )
        if p_peaks_rel.size == 0:
            p_peaks_rel, _ = find_peaks(
                pre_seg, prominence=prom * 0.5, distance=int(0.03 * fs)
            )

        if p_peaks_rel.size > 0:
            p_idx = start + int(p_peaks_rel[-1])
            pr_ms = (r - p_idx) * 1000.0 / fs
            if 50 <= pr_ms <= 300:
                pr_vals.append(pr_ms)

    if not pr_vals:
        return 0.0
    return float(np.mean(pr_vals))


def _extract_segment_features(
    segment: np.ndarray, segment_r_peaks: np.ndarray, segment_idx: int
) -> Dict[str, Any]:
    """
    Full time-domain, HRV, and morphology features per 10 s segment.
    """
    features: Dict[str, Any] = {}

    features["segment_index"] = int(segment_idx)
    features["mean_amplitude"] = float(np.mean(segment))
    features["std_amplitude"] = float(np.std(segment))

    rr_intervals_ms = np.array([])
    if len(segment_r_peaks) >= 2:
        rr_samples = np.diff(segment_r_peaks)
        rr_intervals_ms = rr_samples * 1000.0 / TARGET_FS

    if rr_intervals_ms.size > 0:
        features["rr_intervals_ms"] = rr_intervals_ms.tolist()
        features["mean_rr"] = float(np.mean(rr_intervals_ms))
        features["mean_hr"] = (
            float(60.0 / (features["mean_rr"] / 1000.0))
            if features["mean_rr"] > 0
            else 0.0
        )
        features["SDNN"] = float(np.std(rr_intervals_ms))
        diff_rr = np.diff(rr_intervals_ms)
        features["RMSSD"] = (
            float(np.sqrt(np.mean(diff_rr**2))) if diff_rr.size > 0 else 0.0
        )
        features["pNN50"] = (
            float(np.sum(np.abs(diff_rr) > 50) / diff_rr.size)
            if diff_rr.size > 0
            else 0.0
        )
    else:
        features["rr_intervals_ms"] = []
        features["mean_rr"] = 0.0
        features["mean_hr"] = 0.0
        features["SDNN"] = 0.0
        features["RMSSD"] = 0.0
        features["pNN50"] = 0.0

    features.update(_calculate_morphology_features(segment, segment_r_peaks))
    features.update(_calculate_frequency_hrv(rr_intervals_ms))
    features.update(_calculate_nonlinear_hrv(rr_intervals_ms))

    # Calculate PR interval from waveform
    pr_interval = _calculate_pr_interval(segment, segment_r_peaks, TARGET_FS)
    features["pr_interval"] = float(pr_interval)
    
    return features


# =========================================================
# Ingestion: process an uploaded file & save features to SQL
# =========================================================

def process_and_save_record(file_path: Path) -> str:
    """
    Process one uploaded ECG JSON file:
      - preprocess
      - R-peaks
      - segment into 10s
      - compute features
      - store in ecg_features_annotatable
    """
    filename_key = str(file_path.relative_to(DATA_ROOT_DIR))

    try:
        raw_signal, original_fs = _load_data_from_json(file_path)
        processed_signal = _preprocess(raw_signal, original_fs)
        r_peaks_all = _r_peak_detection(processed_signal, TARGET_FS)
    except Exception as e:
        raise Exception(f"Processing failed for {filename_key}: {e}")

    conn = None
    try:
        conn = db_service._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ecg_features_annotatable (
                    segment_id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    segment_index INT NOT NULL,
                    segment_start_s FLOAT NOT NULL,
                    segment_duration_s FLOAT NOT NULL,
                    arrhythmia_label VARCHAR(50) DEFAULT NULL,
                    arrhythmia_text_notes TEXT DEFAULT '',
                    r_peaks_in_segment TEXT,
                    features_json JSONB,
                    model_pred_label TEXT,
                    model_pred_probs JSONB,
                    cardiologist_notes TEXT,
                    corrected_by TEXT,
                    corrected_at TIMESTAMP,
                    training_round INT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_segment
                    ON ecg_features_annotatable (filename, segment_index);
                """
            )
        conn.commit()

        n_segments = len(processed_signal) // SEGMENT_LENGTH

        with conn.cursor() as cur:
            for i in range(n_segments):
                start = i * SEGMENT_LENGTH
                end = (i + 1) * SEGMENT_LENGTH
                segment = processed_signal[start:end]

                seg_r_peaks_abs = r_peaks_all[
                    (r_peaks_all >= start) & (r_peaks_all < end)
                ]
                seg_r_peaks_rel = seg_r_peaks_abs - start

                feats = _extract_segment_features(segment, seg_r_peaks_rel, i)
                feats_clean = _sanitize_features(feats)
                rpeaks_str = ",".join(map(str, seg_r_peaks_rel))
                segment_start_s = start / TARGET_FS

                cur.execute(
                    """
                    INSERT INTO ecg_features_annotatable
                    (filename, segment_index, segment_start_s, segment_duration_s,
                     r_peaks_in_segment, features_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (filename, segment_index) DO NOTHING
                    """,
                    (
                        filename_key,
                        i,
                        segment_start_s,
                        SEGMENT_DURATION_S,
                        rpeaks_str,
                        json.dumps(feats_clean),
                    ),
                )
        conn.commit()
        return filename_key

    except psycopg2.Error as e:
        raise Exception(f"Database error during insert: {e}")
    finally:
        if conn:
            conn.close()


# =========================================================
# Utility: load a segment signal from disk for plotting/XAI
# =========================================================

def _load_and_segment_raw_data(relative_path: str, segment_index: int) -> List[float]:
    """
    Load raw ECG segment used for plotting and XAI.

    1. Try ecg_data/<relative_path>
    2. If missing, try input_segments/<relative_path>.json
    """
    # Default location (uploads)
    file_path = DATA_ROOT_DIR / relative_path

    # Fallback for dataset JSONs imported via import_json_segments_to_sql
    if not file_path.exists():
        alt1 = DATASET_JSON_DIR / (relative_path + ".json")
        alt2 = DATASET_JSON_DIR / relative_path  # in case filename already has .json
        if alt1.exists():
            file_path = alt1
        elif alt2.exists():
            file_path = alt2
        else:
            raise FileNotFoundError(
                f"ECG file not found at {file_path} or {alt1} or {alt2}"
            )

    full_signal, original_fs = _load_data_from_json(file_path)
    full_signal = _preprocess(full_signal, original_fs)

    start = segment_index * SEGMENT_LENGTH
    end = (segment_index + 1) * SEGMENT_LENGTH
    segment = full_signal[start:end]
    return segment.tolist()


# =========================================================
# Flask Routes
# =========================================================

@app.route("/")
def index():
    """
    Main dashboard view – loads first segment from SQL.
    """
    row = db_service.fetch_one("SELECT MIN(segment_id) FROM ecg_features_annotatable;")
    first_segment_id = row[0] if row and row[0] else 1

    load_segment_id = request.args.get("load_segment_id", first_segment_id)

    try:
        file_list = sorted([p.name for p in DATA_ROOT_DIR.iterdir() if p.is_file()])
    except Exception:
        file_list = []

    return render_template(
        "index.html",
        file_list=file_list,
        initial_segment_id=int(load_segment_id),
        TARGET_FS=TARGET_FS,
    )


@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    """
    Optional: upload a JSON ECG and immediately index segments into the DB.
    """
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("index"))

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = DATA_ROOT_DIR / filename

    try:
        file.save(filepath)
        filename_key = process_and_save_record(filepath)
        new_segment_id = db_service.get_first_segment_id_by_filename(filename_key)
        if not new_segment_id:
            new_segment_id = 1
        return redirect(f"/?load_segment_id={new_segment_id}")
    except Exception as e:
        return (
            f"ERROR: Processing or Database Insertion Failed: {e}. "
            "Please ensure the JSON file is valid.",
            500,
        )


# =========================================================
# XAI Clinical Explanation Endpoint (Option A)
# =========================================================

@app.route("/api/xai/<int:segment_id>")
def api_xai(segment_id: int):
    """
    Clinical XAI endpoint:
      - loads ECG segment from disk
      - loads features from SQL
      - recomputes PR from waveform for better accuracy
      - uses xai.explain_segment(segment_1d, features) to:
          -> run model prediction
          -> return pred_label, probabilities, explanation
      - also stores model_pred_label, model_pred_probs in SQL
    """
    seg = db_service.get_segment_data(segment_id)
    if not seg:
        return jsonify({"error": "Segment not found"}), 404

    # 1) Load ECG signal for this segment
    raw_signal = seg.get("raw_signal")
    
    if not raw_signal or len(raw_signal) == 0:
        try:
            raw_signal = _load_and_segment_raw_data(
                seg["filename"], seg["segment_index"]
            )
        except Exception as e:
            return jsonify({"error": f"Failed to load ECG segment: {e}"}), 500

    segment_np = np.array(raw_signal, dtype=np.float32)

    # 2) Features from SQL
    features = seg.get("features_json") or {}

    # 3) Recompute PR interval from waveform (using r_peaks_in_segment)
    r_field = seg.get("r_peaks_in_segment", "")
    if isinstance(r_field, str):
        r_str = r_field.strip()
        if r_str:
            r_peaks_arr = np.array(
                [int(x) for x in r_str.split(",") if x.strip().isdigit()],
                dtype=int,
            )
        else:
            r_peaks_arr = np.array([], dtype=int)
    elif isinstance(r_field, list):
        r_peaks_arr = np.array(
            [int(x) for x in r_field if x is not None],
            dtype=int,
        )
    else:
        r_peaks_arr = np.array([], dtype=int)

    try:
        pr_interval_ms = _calculate_pr_interval(segment_np, r_peaks_arr, TARGET_FS)
    except Exception:
        pr_interval_ms = 0.0

    # Put PR in features for XAI rules
    features["pr_interval"] = float(pr_interval_ms)

    # 4) Run the model + explanation
    try:
        print(f"🔍 Calling explain_segment for segment {segment_id}...")
        xai_out = explain_segment(segment_np, features)
        print("✅ explain_segment success")
        pred_label = xai_out.get("pred_label", "Unknown")
        probs = xai_out.get("probabilities", None)
        explanation = xai_out.get("explanation", "")
        saliency = xai_out.get("saliency", [])
    except Exception as e:
        # Model unavailable or incompatible - provide placeholder
        import traceback
        traceback.print_exc()
        print(f"⚠️  XAI unavailable: {e}")
        pred_label = "Model Unavailable"
        probs = []
        explanation = (
            "⚠️ Model prediction unavailable. "
            "The model needs to be retrained with the current 9-class list. "
            "You can still annotate segments manually."
        )
        saliency = []

    # 5) Store model prediction back into SQL (best effort, non-fatal)
    try:
        if probs and len(probs) > 0:
            db_service.save_model_prediction(segment_id, pred_label, probs)
    except Exception as e:
        print("Warning: could not save model prediction:", e)

    return jsonify(
        {
            "pred_label": pred_label,
            "probs": probs,
            "explanation": explanation,
            "saliency": saliency,
            "classes": CLASS_NAMES,
        }
    )


# =========================================================
# Segment Fetch (ECG + Features + Annotation) for Dashboard
# =========================================================

# =========================================================
# HELPER: PR Interval Calculation (Heuristic)
# =========================================================
def _calculate_pr_interval(signal, r_peaks, fs):
    """
    Estimate PR interval by looking for P-wave peak in the 200ms window 
    preceding each R-peak.
    Returns the median PR interval in ms.
    """
    if len(r_peaks) < 2:
        print(f"DEBUG: PR Interval - Not enough R-peaks: {len(r_peaks)}")
        return 0.0
    
    pr_intervals = []
    # Window to search for P-wave: 240ms to 30ms before R-peak
    search_window_ms_start = 240
    search_window_ms_end = 30
    
    search_samples_start = int(search_window_ms_start * fs / 1000)
    search_samples_end = int(search_window_ms_end * fs / 1000)
    
    print(f"DEBUG: PR Interval - FS: {fs}, Window Samples: {search_samples_start} to {search_samples_end}")

    for r_idx in r_peaks:
        if r_idx - search_samples_start < 0:
            continue
            
        # Extract window before R-peak
        window = signal[r_idx - search_samples_start : r_idx - search_samples_end]
        
        if len(window) == 0:
            continue
            
        # Find P-wave peak (max value in window)
        p_peak_relative_idx = np.argmax(window)
        p_peak_val = window[p_peak_relative_idx]
        p_peak_idx = (r_idx - search_samples_start) + p_peak_relative_idx
        
        # Calculate PR interval
        pr_ms = (r_idx - p_peak_idx) * 1000 / fs
        
        # Filter unrealistic values
        # Relaxed range: 30 to 400
        if 30 <= pr_ms <= 400:
            pr_intervals.append(pr_ms)
            
    if not pr_intervals:
        print("DEBUG: PR Interval - No valid intervals found after filtering.")
        # Fallback: try to return something if we have any data, or just 0
        return 0.0
        
    median_pr = float(np.median(pr_intervals))
    print(f"DEBUG: PR Interval - Median: {median_pr:.1f} ms (from {len(pr_intervals)} beats)")
    return median_pr

@app.route("/api/segment/<int:segment_id>")
def get_segment_api(segment_id: int):
    """
    Fetch all necessary info for a specific segment ID:
      - ECG signal
      - basic features (mean HR, PR, QRS width)
      - current arrhythmia label & notes
      - R-peaks
    """
    meta = db_service.get_segment_data(segment_id)
    if not meta:
        return jsonify({"error": "Segment not found"}), 404

    # ECG waveform
    raw_signal = meta.get("raw_signal")
    
    if not raw_signal or len(raw_signal) == 0:
        try:
            raw_signal = _load_and_segment_raw_data(meta["filename"], meta["segment_index"])
        except Exception as e:
            return jsonify({"error": f"Failed to load ECG: {e}"}), 500

    features = meta.get("features_json") or {}
    mean_hr = float(features.get("mean_hr", 0.0))

    # Parse r-peaks from DB
    r_field_db = meta.get("r_peaks_in_segment", "")
    r_peaks_for_frontend = ""
    if isinstance(r_field_db, str):
        r_str = r_field_db.strip()
        if r_str:
            r_peaks_for_frontend = r_str
            r_peaks_arr = np.array(
                [int(x) for x in r_str.split(",") if x.strip().isdigit()],
                dtype=int,
            )
        else:
            r_peaks_arr = np.array([], dtype=int)
    elif isinstance(r_field_db, list):
        r_peaks_arr = np.array(
            [int(x) for x in r_field_db if x is not None],
            dtype=int,
        )
        r_peaks_for_frontend = ",".join(str(int(x)) for x in r_peaks_arr)
    else:
        r_peaks_arr = np.array([], dtype=int)
        r_peaks_for_frontend = ""

    # Recompute PR interval from the segment
    try:
        pr_interval_ms = _calculate_pr_interval(np.array(raw_signal), r_peaks_arr, TARGET_FS)
    except Exception:
        pr_interval_ms = 0.0

    # QRS width from features (robust to None/NaN)
    qrs_mean_ms = 0.0
    qrs_list = features.get("qrs_durations_ms")
    if isinstance(qrs_list, list):
        qrs_clean = []
        for v in qrs_list:
            try:
                if v is None:
                    continue
                val = float(v)
                if not np.isnan(val) and not np.isinf(val):
                    qrs_clean.append(val)
            except Exception:
                continue
        if qrs_clean:
            qrs_mean_ms = float(sum(qrs_clean) / len(qrs_clean))

    return jsonify(
        {
            "segment_id": meta["segment_id"],
            "filename": meta["filename"],
            "segment_index": meta["segment_index"],
            "raw_signal": raw_signal,
            "fs": TARGET_FS,
            "length": SEGMENT_LENGTH,
            "arrhythmia_label": meta.get("arrhythmia_label"),
            "notes": meta.get("arrhythmia_text_notes", ""),
            "features": features,
            "mean_hr": mean_hr,
            "pr_interval": float(pr_interval_ms),
            "qrs_mean_ms": float(qrs_mean_ms),
            "r_peaks": r_peaks_for_frontend,
        }
    )


# =========================================================
# Annotation Save Endpoint
# =========================================================

@app.route("/api/annotate", methods=["POST"])
def annotate_segment():
    """
    Receive cardiologist correction & notes:
      - label (corrected arrhythmia)
      - r_peaks (text)
      - notes
      - corrected_by (user name)
    Stored in SQL using db_service.update_annotation(...).
    """
    data = request.get_json() or {}

    segment_id = data.get("segment_id")
    label = data.get("label")
    r_peaks = data.get("r_peaks")
    notes = data.get("notes", "")
    corrected_by = data.get("corrected_by", "Cardiologist")

    if segment_id is None or label is None or r_peaks is None:
        return jsonify({"error": "Missing annotation data"}), 400

    ok = db_service.update_annotation(segment_id, label, r_peaks, notes, corrected_by)
    if not ok:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Failed to save annotation for Segment {segment_id}",
                }
            ),
            500,
        )

    return jsonify(
        {
            "status": "success",
            "message": f"Annotation saved for Segment {segment_id}",
        }
    )


# =========================================================
# Export Corrected Segments → retraining_data/ (JSON)
# =========================================================

@app.route("/api/export_corrected")
def export_corrected():
    """
    Export corrected SQL segments to retraining_data/ as JSON.
    Uses export_corrected_segments.py (your script).
    """
    try:
        from export_corrected_segments import export_corrected_segments

        export_corrected_segments()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================================
# Next / Previous Navigation
# =========================================================

@app.route("/api/next_segment/<int:segment_id>")
def api_next_segment(segment_id: int):
    """
    Return the next available segment_id after the given one.
    If none, wrap to the minimum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id > %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

            cur.execute("SELECT MIN(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "next": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


@app.route("/api/prev_segment/<int:segment_id>")
def api_prev_segment(segment_id: int):
    """
    Return the previous available segment_id before the given one.
    If none, wrap to the maximum segment_id.
    """
    conn = db_service._connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(segment_id)
                FROM ecg_features_annotatable
                WHERE segment_id < %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

            cur.execute("SELECT MAX(segment_id) FROM ecg_features_annotatable")
            row = cur.fetchone()
            if row and row[0] is not None:
                return jsonify({"ok": True, "prev": int(row[0])})

        return jsonify({"ok": False, "error": "No segments"}), 404
    finally:
        conn.close()


# =========================================================
# Retrain Model Endpoint (Button in UI)
# =========================================================

@app.route("/api/retrain_model", methods=["GET", "POST"])
def api_retrain_model():
    """
    Called by dashboard "Retrain Model Using Corrected Segments" button.

    Pipeline:
      1) export_corrected_segments()  -> retraining_data/
      2) run retrain_model.py         -> outputs/checkpoints/best_model.pth
      3) xai.reset_model()            -> reload new weights on next XAI call
    """
    try:
        # 1) Export corrected segments
        from export_corrected_segments import export_corrected_segments

        export_corrected_segments()

        # 2) Run retraining script (CPU or CUDA handled inside train code)
        # Using train_balanced.py which has Focal Loss and aggressive sampling for class imbalance
        script_path = BASE_DIR / "models_training" / "train_balanced.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR / "models_training")  # Run inside that folder to keep imports simple
        )

        if result.returncode != 0:
            return jsonify(
                {
                    "error": "Retraining script failed",
                    "details": result.stderr,
                }
            )

        # 3) Reload model for XAI
        reset_model()

        return jsonify({"status": "ok", "message": "Model retrained and reloaded."})
    except Exception as e:
        return jsonify({"error": str(e)})


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # Run on 0.0.0.0 so you can view from other machines in LAN if needed
    app.run(host="0.0.0.0", port=5000, debug=True)
