#!/usr/bin/env python3
import wfdb
import json
import numpy as np
from pathlib import Path
from scipy.signal import resample
from tqdm import tqdm

INPUT = Path(r"C:\Users\Lifesigns_LS\Documents\porject\Transformers (2)\Transformers\data\afdb_data")
OUTPUT = Path("converted/AFDB")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEG_LEN = 2500  # 10 sec @ 250 Hz

def convert_record(record_name):
    try:
        record_path = str(INPUT / record_name)
        sig, fields = wfdb.rdsamp(record_path)

        fs = fields.get('fs', 250) if isinstance(fields, dict) else fields.fs
        ecg = sig[:, 0]

        if fs != 250:
            ecg = resample(ecg, int(len(ecg) * 250 / fs))
            fs = 250

        ann = wfdb.rdann(record_path, "atr")

        # AFDB annotation → "AFIB" or normal
        rhythm = {}
        for aos, key in zip(ann.aux_note, ann.sample):
            if "AFIB" in aos:
                rhythm[key] = "Atrial Fibrillation"
            else:
                rhythm[key] = "Sinus Rhythm"

        n_segments = len(ecg) // SEG_LEN

        for i in range(n_segments):
            start = i * SEG_LEN
            end = start + SEG_LEN
            seg = ecg[start:end].tolist()

            seg_labels = [rhythm[k] for k in rhythm if start <= k < end]
            label = max(seg_labels, key=seg_labels.count) if seg_labels else "Sinus Rhythm"

            out = {
                "ECG_CH_A": seg,
                "fs": 250,
                "label": label,
                "dataset": "AFDB",
                "record": record_name,
                "segment_index": i
            }

            fname = OUTPUT / f"AFDB__{record_name}_seg_{i:04d}.json"
            fname.write_text(json.dumps(out))
        
        return n_segments
        
    except FileNotFoundError:
        print(f"⚠️  Record {record_name} not found")
        return 0
    except Exception as e:
        print(f"❌ Error processing {record_name}: {e}")
        return 0

def main():
    records = ["04015", "04043", "04048", "04126", "04746"]
    total_segments = 0
    
    print(f"🔄 Converting {len(records)} AFDB records to JSON...")
    
    for rec in tqdm(records, desc="Records", unit="rec"):
        segments = convert_record(rec)
        total_segments += segments
    
    print(f"\n✅ Conversion complete!")
    print(f"   • Records: {len(records)}")
    print(f"   • Total segments: {total_segments}")
    print(f"   • Output: {OUTPUT}")

if __name__ == "__main__":
    main()
