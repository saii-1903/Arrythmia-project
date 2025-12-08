# src/db_loader.py
import json
from typing import List, Dict, Any, Optional
from db_service import PSQL_CONN_PARAMS
import psycopg2

def _connect():
    return psycopg2.connect(**PSQL_CONN_PARAMS)

def fetch_annotated_segments(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Return list of dict rows from ecg_features_annotatable:
      segment_id, filename, segment_index, features_json (dict), arrhythmia_label, r_peaks_in_segment
    Fail-safe: returns [] on error.
    """
    try:
        conn = _connect()
    except Exception as ex:
        print(f"[db_loader] WARNING: DB connect failed: {ex}")
        return []

    out = []
    try:
        with conn.cursor() as cur:
            q = """
                SELECT segment_id, filename, segment_index, features_json, arrhythmia_label, r_peaks_in_segment
                FROM ecg_features_annotatable
            """
            if limit:
                q += f" LIMIT {int(limit)}"
            cur.execute(q)
            rows = cur.fetchall()
            for seg_id, filename, seg_idx, features_json, arr_label, r_peaks in rows:
                # normalize features_json
                if isinstance(features_json, str):
                    try:
                        features = json.loads(features_json)
                    except:
                        features = {}
                else:
                    features = features_json or {}
                # normalize r_peaks
                if isinstance(r_peaks, str):
                    try:
                        rp = [int(x) for x in r_peaks.split(',') if x.strip()!='']
                    except:
                        rp = []
                elif isinstance(r_peaks, (list, tuple)):
                    rp = list(r_peaks)
                else:
                    rp = []
                out.append({
                    "segment_id": seg_id,
                    "filename": filename,
                    "segment_index": int(seg_idx) if seg_idx is not None else 0,
                    "features_json": features,
                    "arrhythmia_label": arr_label,
                    "r_peaks_in_segment": rp
                })
    except Exception as ex:
        print(f"[db_loader] WARNING: query failed: {ex}")
    finally:
        try:
            conn.close()
        except:
            pass
    return out
