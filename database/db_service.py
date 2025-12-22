import psycopg2
import json
from typing import List, Dict, Any

# ---------------------------------------
# PostgreSQL Connection Settings
# ---------------------------------------
PSQL_CONN_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",         # <-- your password
    "host": "127.0.0.1",
    "port": "5432"
}

def _connect():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(**PSQL_CONN_PARAMS)

# =====================================================================
# FETCH LIST OF FILES
# =====================================================================
def get_segment_list() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT filename,
                   COUNT(*) AS segment_count,
                   SUM(CASE WHEN arrhythmia_label IS NULL
                               OR arrhythmia_label='Unlabeled'
                            THEN 1 ELSE 0 END) AS unlabeled_count
            FROM ecg_features_annotatable
            GROUP BY filename
            ORDER BY filename;
        """)

        rows = cur.fetchall()
        return [
            {
                "filename": r[0],
                "segment_count": r[1],
                "unlabeled_count": r[2]
            }
            for r in rows
        ]
    except:
        return []
    finally:
        if conn:
            conn.close()

# =====================================================================
# FETCH A SINGLE SEGMENT
# =====================================================================
def get_segment_data(segment_id: int):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    segment_id,
                    filename,
                    segment_index,
                    segment_start_s,
                    segment_duration_s,
                    arrhythmia_label,
                    arrhythmia_text_notes,
                    r_peaks_in_segment,
                    features_json,
                    cardiologist_notes,
                    corrected_by,
                    corrected_at,
                    training_round,
                    raw_signal,
                    pr_interval,
                    segment_fs,
                    dataset_source
                FROM ecg_features_annotatable
                WHERE segment_id = %s
                """,
                (segment_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            cols = [
                "segment_id",
                "filename",
                "segment_index",
                "segment_start_s",
                "segment_duration_s",
                "arrhythmia_label",
                "arrhythmia_text_notes",
                "r_peaks_in_segment",
                "features_json",
                "cardiologist_notes",
                "corrected_by",
                "corrected_at",
                "training_round",
                "raw_signal",
                "pr_interval",
                "segment_fs",
                "dataset_source",
            ]

            data = {cols[i]: row[i] for i in range(len(cols))}
            return data

    except Exception as e:
        print("DB ERROR get_segment_data:", e)
        return None
    finally:
        conn.close()


# =====================================================================
# UPDATE SEGMENT ANNOTATION
# =====================================================================
def update_annotation(segment_id: int, label: str, r_peaks, notes: str, corrected_by: str = "Cardiologist") -> bool:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        r_str = ",".join(map(str, r_peaks)) if r_peaks else ""

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET arrhythmia_label = %s,
                r_peaks_in_segment = %s,
                arrhythmia_text_notes = %s,
                corrected_by = %s,
                corrected_at = CURRENT_TIMESTAMP
            WHERE segment_id = %s;
        """, (label, r_str, notes, corrected_by, segment_id))

        conn.commit()
        return cur.rowcount > 0

    except Exception as e:
        print("DB ERROR update_annotation:", e)
        return False
    finally:
        if conn:
            conn.close()

# =====================================================================
# SAVE MODEL PREDICTION (for XAI UI)
# =====================================================================
def save_model_prediction(segment_id: int, pred_label: str, probs_list):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            UPDATE ecg_features_annotatable
            SET model_pred_label = %s,
                model_pred_probs = %s
            WHERE segment_id = %s;
        """, (pred_label, json.dumps(probs_list), segment_id))

        conn.commit()

    except Exception as e:
        print("DB ERROR save_model_prediction:", e)
    finally:
        if conn:
            conn.close()
    return True
# =====================================================================
# FIND FIRST SEGMENT WITH raw_signal
# =====================================================================
def get_min_segment_id_with_signal() -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT MIN(segment_id)
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL;
        """)

        row = cur.fetchone()
        return int(row[0]) if row and row[0] else 0

    except Exception as e:
        print("DB ERROR get_min_segment_id_with_signal:", e)
        return 0
    finally:
        if conn:
            conn.close()

# =====================================================================
# GENERIC fetch_one() used by your app.py
# =====================================================================
def fetch_one(sql: str, params=None):
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception as e:
        print("DB fetch_one error:", e)
        return None
    finally:
        if conn:
            conn.close()

# =====================================================================
# Find first segment for a newly uploaded JSON
# =====================================================================
def get_first_segment_id_by_filename(filename_key: str) -> int:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT segment_id
            FROM ecg_features_annotatable
            WHERE filename = %s
            ORDER BY segment_index ASC
            LIMIT 1;
        """, (filename_key,))

        row = cur.fetchone()
        return row[0] if row else 0

    except Exception as e:
        print("DB ERROR get_first_segment_id_by_filename:", e)
        return 0

    finally:
        if conn:
            conn.close()

# =====================================================================
# GET ALL CORRECTED SEGMENTS (For Export)
# =====================================================================
def get_all_corrected() -> List[Dict[str, Any]]:
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT segment_id,
                   filename,
                   segment_index,
                   arrhythmia_label,
                   model_pred_label,
                   features_json,
                   raw_signal,
                   segment_fs,
                   dataset_source
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
              AND arrhythmia_label IS NOT NULL
              AND arrhythmia_label != 'Unlabeled';
        """)
        
        rows = cur.fetchall()
        cols = [
            "segment_id", "filename", "segment_index", "arrhythmia_label",
            "model_pred_label", "features_json", "raw_signal", "segment_fs", "dataset_source"
        ]
        
        results = []
        for r in rows:
            results.append({cols[i]: r[i] for i in range(len(cols))})
            
        return results

    except Exception as e:
        print("DB ERROR get_all_corrected:", e)
        return []
    finally:
        if conn:
            conn.close()
