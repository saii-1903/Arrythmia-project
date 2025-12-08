#!/usr/bin/env python3
"""
Export corrected ECG segments from PostgreSQL into retraining_data/ as JSON.

A "corrected" segment is one where:
- arrhythmia_label IS NOT NULL
- OR model_pred_label != arrhythmia_label
- OR cardiologist_notes IS NOT NULL

These JSONs will later be appended to the main dataset for retraining.
"""

import os
import json
import psycopg2
from pathlib import Path

EXPORT_DIR = Path("retraining_data")
EXPORT_DIR.mkdir(exist_ok=True)

# ------------------------ PostgreSQL Connection ------------------------
conn = psycopg2.connect(
    host="localhost",
    database="your_db_name",
    user="your_username",
    password="your_password"
)
conn.autocommit = True
cur = conn.cursor()

# -----------------------------------------------------------------------

SQL_QUERY = """
SELECT 
    segment_id,
    filename,
    segment_index,
    raw_signal,
    features_json,
    pr_interval,
    arrhythmia_label,
    model_pred_label,
    model_pred_probs,
    cardiologist_notes,
    segment_fs,
    dataset_source
FROM ecg_features_annotatable
WHERE 
    arrhythmia_label IS NOT NULL 
    OR cardiologist_notes IS NOT NULL 
    OR (model_pred_label IS NOT NULL AND model_pred_label != arrhythmia_label);
"""

def main():
    cur.execute(SQL_QUERY)
    rows = cur.fetchall()

    print(f"Found {len(rows)} corrected segments to export.")

    for row in rows:
        (
            seg_id, 
            fname, 
            seg_idx, 
            raw_signal, 
            features,
            pr_interval,
            arr_label,
            model_label,
            probs,
            notes,
            fs,
            dataset
        ) = row

        js = {
            "segment_id": seg_id,
            "filename": fname,
            "segment_index": seg_idx,
            "ECG_CH_A": raw_signal,
            "fs": fs if fs else 250,
            "features": features,
            "pr_interval": pr_interval,
            "arrhythmia_label": arr_label,
            "model_pred_label": model_label,
            "model_pred_probs": probs,
            "notes": notes,
            "dataset": dataset,
        }

        out_path = EXPORT_DIR / f"SQL_corrected_seg_{seg_id}.json"
        out_path.write_text(json.dumps(js, indent=2))

    print(f"Export complete. JSON files saved under {EXPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
