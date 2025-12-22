"""
data_loader.py
--------------
Loads ECG JSON dataset for training CNN+Transformer model.

✔ Reads all *.json files inside dataset folder
✔ Normalizes label text to CLASS_NAMES
✔ Converts to numpy arrays suitable for collate_fn
✔ Handles corrupted JSON files gracefully
✔ Ensures every signal = 250 Hz, 10 sec (2500 samples)
"""

import json
import numpy as np
import psycopg2
from pathlib import Path
from scipy.signal import resample

# ============================================================
# ============================================================
# ============================================================
# FINAL FIXED CLASS LIST (COMPREHENSIVE + COMBINATIONS)
# ============================================================

CLASS_NAMES = [
    # 0-21: Standard
    "Sinus Rhythm",                  # 0
    "Sinus Bradycardia",             # 1
    "Sinus Tachycardia",             # 2
    "Supraventricular Tachycardia",  # 3
    "Atrial Fibrillation",           # 4
    "Atrial Flutter",                # 5
    "Junctional Rhythm",             # 6
    "Idioventricular Rhythm",        # 7
    "Ventricular Tachycardia",       # 8
    "Ventricular Fibrillation",      # 9
    "1st Degree AV Block",           # 10
    "2nd Degree AV Block Type 1",    # 11 
    "2nd Degree AV Block Type 2",    # 12 
    "3rd Degree AV Block",           # 13
    "PVCs",                          # 14
    "PVC Bigeminy",                  # 15
    "PVC Trigeminy",                 # 16
    "PVC Couplet",                   # 17
    "PAC",                           # 18
    "PAC Bigeminy",                  # 19
    "Bundle Branch Block",           # 20
    "Artifact",                      # 21
    
    # NEW COMBINATIONS
    "Sinus Bradycardia + PVC",       # 22
    "Sinus Tachycardia + PVC",       # 23
    "Sinus Bradycardia + PAC",       # 24
    "Sinus Tachycardia + PAC",       # 25
    "Atrial Fibrillation + PVC",     # 26
    "Atrial Flutter + PVC",          # 27
    "1st Degree AV Block + PVC",     # 28
    "Sinus Bradycardia + PVC Bigeminy", # 29
    "Sinus Tachycardia + PVC Bigeminy", # 30
    
    # COMPLEX PATTERNS (Rules-Based / Advanced)
    "Atrial Couplet",                # 31
    "Atrial Run",                    # 32
    "Ventricular Run",               # 33
    "NSVT",                          # 34
    "PSVT",                          # 35
    "Pause",                         # 36
]

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

TARGET_FS = 250
SEG_LEN = TARGET_FS * 10 

# ============================================================
# SIMPLE LABEL NORMALIZATION
# ============================================================

LABEL_MAP = {
    # Normals
    "NORMAL": "Sinus Rhythm", "NSR": "Sinus Rhythm", "NORM": "Sinus Rhythm",
    "SB": "Sinus Bradycardia", "BRADY": "Sinus Bradycardia", "SINUS BRADYCARDIA": "Sinus Bradycardia",
    "ST": "Sinus Tachycardia", "TACHY": "Sinus Tachycardia", "SINUS TACHYCARDIA": "Sinus Tachycardia",
    
    # SVT / Atrial
    "SVT": "Supraventricular Tachycardia", 
    "AF": "Atrial Fibrillation", "AFIB": "Atrial Fibrillation", "ATRIAL FIBRILLATION": "Atrial Fibrillation",
    "AFL": "Atrial Flutter", "ATRIAL FLUTTER": "Atrial Flutter",
    
    # Junctional
    "JUNCTIONAL": "Junctional Rhythm", 
    
    # Ventricular
    "IVR": "Idioventricular Rhythm",
    "VT": "Ventricular Tachycardia", 
    "VF": "Ventricular Fibrillation", 
    
    # Blocks
    "1AVB": "1st Degree AV Block", "1' AV BLOCK": "1st Degree AV Block", 
    "WENCKEBACH": "2nd Degree AV Block Type 1", 
    "MOBITZ II": "2nd Degree AV Block Type 2", 
    "3AVB": "3rd Degree AV Block", 
    "BBB": "Bundle Branch Block", "LBBB": "Bundle Branch Block", "RBBB": "Bundle Branch Block",
    
    # Ectopy
    "PVC": "PVCs", "VPB": "PVCs",
    "PVC BIGEMINY": "PVC Bigeminy", 
    "PVC TRIGEMINY": "PVC Trigeminy", 
    "PVC COUPLET": "PVC Couplet", 
    "PAC": "PAC", 
    "PAC BIGEMINY": "PAC Bigeminy", 
    
    "ARTIFACT": "Artifact"
}


def normalize_label(label: str):
    """Convert any dataset label into one of the comprehensive classes."""
    if label is None: return "Sinus Rhythm"
    if not isinstance(label, str): label = str(label)

    L = label.strip().upper()

    # Direct passthrough if already correct
    for c in CLASS_NAMES:
        if c.upper() == L:
            return c
    
    # Also Check exact map
    if L in LABEL_MAP: return LABEL_MAP[L]
    
    # Heuristic Fallbacks
    if "WENCKEBACH" in L: return "2nd Degree AV Block Type 1"
    if "MOBITZ" in L: return "2nd Degree AV Block Type 2"
    if "BIGEMINY" in L: 
        return "PVC Bigeminy" if "PVC" in L or "VENTRICULAR" in L else "PAC Bigeminy"
    if "FLUTTER" in L: return "Atrial Flutter"
    if "FIBRILLATION" in L:
        return "Ventricular Fibrillation" if "VENTRICULAR" in L else "Atrial Fibrillation"
    
    return "Sinus Rhythm" # Default fallback


# ============================================================
# DATASET - Lazy and robust JSON reading
# ============================================================

class ECGDataset:
    """
    Lightweight dataset that lists JSON files at init and reads them on demand.
    This avoids long startup times when many JSONs exist.
    """

    def __init__(self, data_dir):
        """
        data_dir: folder (string or Path) containing JSON ECG segments
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise RuntimeError(f"Dataset folder not found: {self.data_dir}")

        # only top-level *.json (user previously used many folder layouts; this keeps it simple)
        self.files = sorted(list(self.data_dir.glob("*.json")))

        if len(self.files) == 0:
            raise RuntimeError(f"No JSON files found in dataset: {data_dir}")

        print(f"[Dataset] Found {len(self.files)} JSON ECG segments in {self.data_dir}.")

    def __len__(self):
        return len(self.files)

    def _safe_load_json(self, fpath: Path):
        try:
            with fpath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data
        except Exception as e:
            # Print a short message; don't crash on one corrupted file
            print(f"[WARN] Failed to load JSON {fpath.name}: {e}")
            return None

    def _extract_signal_and_fs(self, data: dict):
        # Try typical locations (mirrors your earlier loader)
        if data is None:
            return None, None

        # 1) Standard our-converted format
        if "ECG_CH_A" in data and data.get("ECG_CH_A") is not None:
            sig = np.array(data["ECG_CH_A"], dtype=np.float32)
            fs = int(data.get("fs", TARGET_FS))
            return sig, fs

        # 2) SensorData from raw Lifesigns / PTB / etc
        sd = data.get("SensorData")
        if (
            isinstance(sd, list)
            and len(sd) > 0
            and isinstance(sd[0], dict)
            and "ECG_CH_A" in sd[0]
        ):
            sig = np.array(sd[0]["ECG_CH_A"], dtype=np.float32)
            fs = int(data.get("fs", sd[0].get("fs", TARGET_FS)))
            return sig, fs

        # 3) Sometimes wrapped inside features_json / meta
        fj = data.get("features_json")
        if isinstance(fj, dict) and "segment_signal" in fj:
            sig = np.array(fj["segment_signal"], dtype=np.float32)
            fs = int(fj.get("fs", data.get("fs", TARGET_FS)))
            return sig, fs

        meta = data.get("meta")
        if isinstance(meta, dict) and "segment_signal" in meta:
            sig = np.array(meta["segment_signal"], dtype=np.float32)
            fs = int(meta.get("fs", TARGET_FS))
            return sig, fs

        # 4) last resort: generic 'signal'
        if "signal" in data:
            try:
                sig = np.array(data["signal"], dtype=np.float32)
                fs = int(data.get("fs", TARGET_FS))
                return sig, fs
            except Exception:
                pass

        return None, None

    def _resample_and_fixlen(self, sig, orig_fs):
        # If orig_fs invalid, assume TARGET_FS
        try:
            orig_fs = int(orig_fs)
        except Exception:
            orig_fs = TARGET_FS

        if orig_fs != TARGET_FS and len(sig) > 1:
            # simple resample using scipy.signal.resample
            try:
                new_len = int(len(sig) * float(TARGET_FS) / float(orig_fs))
                sig = resample(sig, new_len).astype(np.float32)
            except Exception:
                # fallback: numpy interp
                idx_old = np.arange(len(sig))
                idx_new = np.linspace(
                    0, len(sig) - 1,
                    int(len(sig) * float(TARGET_FS) / float(orig_fs))
                )
                sig = np.interp(idx_new, idx_old, sig).astype(np.float32)

        # pad/truncate to SEG_LEN
        if len(sig) < SEG_LEN:
            pad = SEG_LEN - len(sig)
            sig = np.pad(sig, (0, pad))
        elif len(sig) > SEG_LEN:
            sig = sig[:SEG_LEN]

        return sig.astype(np.float32)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = self._safe_load_json(fpath)

        if data is None:
            # return a zero sample so training doesn't crash; label 0 is Sinus Rhythm
            return {
                "signal": np.zeros(SEG_LEN, dtype=np.float32),
                "label": 0,
                "meta": {"source": str(fpath)},
            }

        sig, fs = self._extract_signal_and_fs(data)
        if sig is None:
            # fallback: zero sample
            return {
                "signal": np.zeros(SEG_LEN, dtype=np.float32),
                "label": 0,
                "meta": {"source": str(fpath)},
            }

        sig = self._resample_and_fixlen(sig, fs)

        # LABEL resolution
        label_txt = None
        if data.get("label"):
            label_txt = data.get("label")
        elif isinstance(data.get("features_json"), dict) and data["features_json"].get("inferred_label"):
            label_txt = data["features_json"].get("inferred_label")
        elif isinstance(data.get("meta"), dict) and data["meta"].get("arrhythmia_label"):
            label_txt = data["meta"].get("arrhythmia_label")

        # normalize and map to index
        label_norm = normalize_label(label_txt or "Sinus Rhythm")
        y = CLASS_INDEX.get(label_norm, 0)

        meta = data.get("meta", {"source": str(fpath)})
        return {"signal": sig, "label": int(y), "meta": meta}


# ============================================================
# SQL DATASET
# ============================================================

class ECGRawDatasetSQL:
    def __init__(self, limit=None):
        self.conn_params = {
            "dbname": "ecg_analysis",
            "user": "ecg_user",
            "password": "sais",
            "host": "127.0.0.1",
            "port": "5432"
        }
        self.samples = []  # [(segment_id, label_int), ...]
        self._load_metadata(limit)

    def _connect(self):
        return psycopg2.connect(**self.conn_params)

    def _load_metadata(self, limit):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT segment_id, arrhythmia_label
                    FROM ecg_features_annotatable
                    WHERE raw_signal IS NOT NULL
                      AND arrhythmia_label IS NOT NULL
                      AND arrhythmia_label != 'Unlabeled'
                """
                if limit:
                    query += f" LIMIT {limit}"
                cur.execute(query)
                rows = cur.fetchall()

                count = 0
                for r in rows:
                    seg_id, lbl_str = r
                    # Validate label
                    if not lbl_str: 
                        continue
                    
                    # Normalize
                    lbl_norm = normalize_label(lbl_str)
                    lbl_idx = CLASS_INDEX.get(lbl_norm, 0)
                    
                    self.samples.append((seg_id, lbl_idx))
                    count += 1
                
                print(f"[ECGRawDatasetSQL] Loaded {count} segments from DB.")
        finally:
            conn.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg_id, label_idx = self.samples[idx]
        
        # Fetch signal
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT raw_signal, segment_fs FROM ecg_features_annotatable WHERE segment_id = %s", (seg_id,))
                row = cur.fetchone()
                if not row:
                    # Should not happen if metadata is consistent
                    return {"signal": np.zeros(SEG_LEN, dtype=np.float32), "label": label_idx, "meta": {"id": seg_id}}
                
                raw_sig, fs = row
                # raw_sig is likely a list or array from PG
                sig = np.array(raw_sig, dtype=np.float32)
                
                if fs is None: fs = TARGET_FS
                fs = int(fs)
                
                # Resample / Fix Len using the existing logic
                # We can reuse logic or implement here. 
                # Since ECGRawDatasetSQL is separate, we'll duplicate the helper or make it static.
                # Re-using the helper from ECGDataset class is hard unless we refactor.
                # I'll implement a simple static version or inline it.
                
                # Inline resample/fixlen logic
                if fs != TARGET_FS and len(sig) > 1:
                    new_len = int(len(sig) * float(TARGET_FS) / float(fs))
                    sig = resample(sig, new_len).astype(np.float32)
                
                if len(sig) < SEG_LEN:
                    pad = SEG_LEN - len(sig)
                    sig = np.pad(sig, (0, pad))
                elif len(sig) > SEG_LEN:
                    sig = sig[:SEG_LEN]
                
                return {
                    "signal": sig, 
                    "label": int(label_idx), 
                    "meta": {"id": seg_id}
                }
        finally:
            conn.close()

