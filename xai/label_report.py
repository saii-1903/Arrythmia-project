# tools/label_report.py
import argparse
from pathlib import Path
import json
from collections import Counter
from data_loader import _map_label_text_to_int, CLASS_NAMES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()
    folder = Path(args.folder)
    files = list(folder.glob("*.json"))
    counts = Counter()
    for f in files:
        try:
            js = json.loads(f.read_text(encoding="utf-8"))
        except:
            continue
        txt = js.get("label") or js.get("features_json", {}).get("inferred_label")
        cls = _map_label_text_to_int(txt)
        counts[cls] += 1
    print("=== LABEL COUNTS ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i:02d} {name:35} -> {counts[i]}")
    missing = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if counts[i]==0]
    if missing:
        print("MISSING:", missing)
    else:
        print("All classes present (maybe with low counts).")

if __name__ == "__main__":
    main()
