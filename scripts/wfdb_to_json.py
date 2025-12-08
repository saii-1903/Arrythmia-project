#!/usr/bin/env python3
import wfdb
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

INPUT = Path(r"data/mitdb_data")
OUTPUT = Path(r"C:\Users\Lifesigns_LS\Documents\porject\Transformers (2)\Transformers\converted\mitdb")
OUTPUT.mkdir(parents=True, exist_ok=True)

MITDB_LABEL_MAP = {
    "N": "Sinus Rhythm",
    "A": "Atrial Fibrillation",
    "V": "PVCs",
    "/": "Supraventricular Tachycardia"
}

SEG_LEN = 2500  # 10 sec * 250 Hz

def convert_record(record_name):
    try:
        record_path = str(INPUT / record_name)
        sig, fields = wfdb.rdsamp(record_path)
        ann = wfdb.rdann(record_path, "atr")

        ecg = sig[:, 0]  # channel A
        fs = fields.get('fs', 250) if isinstance(fields, dict) else fields.fs
        
        if fs != 250:
            # resample to 250
            from scipy.signal import resample
            ecg = resample(ecg, int(len(ecg) * 250 / fs))
            fs = 250

        # assign per-beat labels → convert to segment labels
        beat_labels = {}
        for idx, sym in zip(ann.sample, ann.symbol):
            beat_labels[idx] = MITDB_LABEL_MAP.get(sym, "Sinus Rhythm")

        n_segments = len(ecg) // SEG_LEN
        
        for i in range(n_segments):
            start = i * SEG_LEN
            end = start + SEG_LEN
            seg = ecg[start:end].tolist()

            seg_beats = [beat_labels[s] for s in beat_labels if start <= s < end]

            if seg_beats:
                label = max(seg_beats, key=seg_beats.count)
            else:
                label = "Sinus Rhythm"

            out = {
                "ECG_CH_A": seg,
                "fs": 250,
                "label": label,
                "dataset": "MITDB",
                "record": record_name,
                "segment_index": i
            }

            fname = OUTPUT / f"MITDB__{record_name}_seg_{i:04d}.json"
            fname.write_text(json.dumps(out))
        
        return n_segments
        
    except FileNotFoundError:
        print(f"⚠️  Record {record_name} not found")
        return 0
    except Exception as e:
        print(f"❌ Error processing {record_name}: {e}")
        return 0

def main():
    records = ["100", "101", "102", "103", "104", "105", "106", "107", "108"]
    total_segments = 0
    
    print(f"🔄 Converting {len(records)} MITDB records to JSON...")
    
    for rec in tqdm(records, desc="Records", unit="rec"):
        segments = convert_record(rec)
        total_segments += segments
    
    print(f"\n✅ Conversion complete!")
    print(f"   • Records: {len(records)}")
    print(f"   • Total segments: {total_segments}")
    print(f"   • Output: {OUTPUT}")

if __name__ == "__main__":
    main()
