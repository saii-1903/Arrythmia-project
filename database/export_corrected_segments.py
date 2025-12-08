#!/usr/bin/env python3
"""
export_corrected_segments.py

Export corrected segments from PostgreSQL to JSON files
under retraining_data/ folder, in the same schema used
by ECGRawDataset & training:

{
  "ECG_CH_A": [...],
  "fs": 250,
  "label": "Atrial Fibrillation",
  "dataset": "SQL_CORRECTED",
  "record": "AFDB__04015",
  "segment_index": 41,
  "features": {...}
}
"""

from pathlib import Path
import json
import db_service
import numpy as np

OUTPUT_DIR = Path("retraining_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_corrected_segments():
    rows = db_service.get_all_corrected()
    print(f"Exporting {len(rows)} corrected segments to {OUTPUT_DIR}...")

    for row in rows:
        seg_id = row["segment_id"]
        filename = row["filename"]
        segment_index = row["segment_index"]
        label = row["arrhythmia_label"] or row["model_pred_label"] or "Sinus Rhythm"
        features = row["features_json"] or {}
        raw_signal = row.get("raw_signal") or []

        # raw_signal is stored as PostgreSQL array; convert to list
        if isinstance(raw_signal, list):
            sig = raw_signal
        else:
            # fallback: if stored as string or None
            try:
                sig = list(raw_signal)
            except Exception:
                sig = []

        js = {
            "ECG_CH_A": sig,
            "fs": row.get("segment_fs", 250),
            "label": label,
            "dataset": row.get("dataset_source", "SQL_CORRECTED"),
            "record": filename,
            "segment_index": segment_index,
            "features": features
        }

        out_name = OUTPUT_DIR / f"SQLCORR__seg_{seg_id:06d}.json"
        out_name.write_text(json.dumps(js), encoding="utf-8")

    print("Done export_corrected_segments().")

if __name__ == "__main__":
    export_corrected_segments()
