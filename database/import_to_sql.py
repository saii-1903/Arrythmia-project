#!/usr/bin/env python3
"""
Clean SQL importer for JSON ECG segments.
✔ Normalizes labels to match training classes
✔ Stores true filename
✔ Computes PR, RR, QRS safely
✔ Cleans JSON for SQL
"""

import os
import json
import numpy as np
import psycopg2
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------
# CONFIG
# ----------------------------------------

JSON_FOLDER = Path(r"C:\Users\Lifesigns_LS\Documents\porject\Transformers (2)\Transformers\merged_data")
FS = 250

conn = psycopg2.connect(
    host="localhost",
    database="ecg_analysis",
    user="ecg_user",
    password="sais"
)
conn.autocommit = True
cur = conn.cursor()

# ----------------------------------------
# Import normalize_label from data_loader
# ----------------------------------------
from data_loader import normalize_label   # ⬅ IMPORTANT


# ----------------------------------------
# Helpers
# ----------------------------------------

def clean_json(obj):
    """Recursively replace NaN/inf with None."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, list):
        return [clean_json(x) for x in obj]

    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}

    return obj


# ---------------------------
# SQL INSERT
# ---------------------------

INSERT_SQL = """
INSERT INTO ecg_features_annotatable
(filename, segment_index, raw_signal, features_json, pr_interval, arrhythmia_label)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING;
"""


# ----------------------------------------
# Feature extraction
# ----------------------------------------

def compute_features(signal, fs=250):
    try:
        sig_clean = nk.ecg_clean(signal, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(sig_clean, sampling_rate=fs)
        r_locs = np.array(rpeaks.get("ECG_R_Peaks", []), dtype=int)

        # RR intervals
        if len(r_locs) >= 2:
            rr = np.diff(r_locs) * (1000 / fs)
        else:
            rr = []

        # QRS durations
        try:
            _, waves = nk.ecg_delineate(sig_clean, r_locs, sampling_rate=fs)
            qrs = []
            for q_on, r_off in zip(waves.get("ECG_R_Onsets", []), waves.get("ECG_R_Offsets", [])):
                if q_on is not None and r_off is not None:
                    qrs.append((r_off - q_on) * (1000 / fs))
        except Exception:
            qrs = []

        # Mean HR
        try:
            hr = nk.ecg_rate(r_locs, sampling_rate=fs)
            mean_hr = float(np.mean(hr)) if len(hr) else 0.0
        except Exception:
            mean_hr = 0.0

        return {
            "r_peaks": r_locs.tolist(),
            "rr_intervals_ms": [float(x) for x in rr],
            "qrs_durations_ms": [float(x) for x in qrs],
            "mean_hr": mean_hr,
        }

    except Exception:
        return {
            "r_peaks": [],
            "rr_intervals_ms": [],
            "qrs_durations_ms": [],
            "mean_hr": 0.0,
        }


def compute_pr_interval(signal, r_locs, fs=250):
    try:
        if len(r_locs) < 1:
            return None

        _, waves = nk.ecg_delineate(signal, r_locs, sampling_rate=fs)
        p_on = waves.get("ECG_P_Onsets", [])
        r_on = waves.get("ECG_R_Onsets", [])

        pr = []
        for p, r in zip(p_on, r_on):
            if p is not None and r is not None:
                pr.append((r - p) * (1000 / fs))

        return float(np.mean(pr)) if len(pr) else None

    except Exception:
        return None


# ----------------------------------------
# MAIN IMPORT LOOP
# ----------------------------------------

def main():
    files = sorted(JSON_FOLDER.glob("*.json"))
    print(f"Found {len(files)} JSON segments.")

    for js_file in tqdm(files, desc="Importing"):
        try:
            data = json.loads(js_file.read_text())

            signal = np.array(data["ECG_CH_A"], dtype=float)
            seg_idx = int(data.get("segment_index", 0))

            # ------------------------------
            # NORMALIZE LABEL HERE
            # ------------------------------
            raw_label = data.get("label", None)
            label = normalize_label(raw_label)   # ⭐ fixed
            # ------------------------------

            filename = js_file.name  # MUST store exact filename

            # Extract features
            feats = compute_features(signal, fs=FS)
            pr = compute_pr_interval(signal, feats["r_peaks"], fs=FS)

            feats_clean = clean_json(feats)
            pr_clean = None if pr is None else float(pr)

            # SQL insert
            cur.execute(
                INSERT_SQL,
                (
                    filename,
                    seg_idx,
                    signal.tolist(),
                    json.dumps(feats_clean),
                    pr_clean,
                    label
                )
            )

        except Exception as e:
            print(f"ERROR in {js_file.name}: {e}")

    print("✔ SQL import completed.")


if __name__ == "__main__":
    main()
