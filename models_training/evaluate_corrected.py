
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import ECGRawDatasetSQL, CLASS_NAMES, CLASS_INDEX, normalize_label
from models import CNNTransformerClassifier

class CorrectedSegmentsDataset(ECGRawDatasetSQL):
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
                      AND corrected_by IS NOT NULL
                    ORDER BY corrected_at DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                cur.execute(query)
                rows = cur.fetchall()

                count = 0
                for r in rows:
                    seg_id, lbl_str = r
                    if not lbl_str: continue
                    
                    lbl_norm = normalize_label(lbl_str)
                    lbl_idx = CLASS_INDEX.get(lbl_norm, 0)
                    
                    self.samples.append((seg_id, lbl_idx))
                    count += 1
                
                print(f"[CorrectedDataset] Loaded {count} corrected segments from DB (Ordered by recency).")
        finally:
            conn.close()

def evaluate():
    print("=== Evaluation on CORRECTED Segments ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        dataset = CorrectedSegmentsDataset(limit=300)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        print("No corrected segments found in SQL database.")
        print("Tip: Annotate some segments in the dashboard first.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    ckpt_path = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading model from {ckpt_path}...")
    try:
        model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
        state = torch.load(ckpt_path, map_location=device)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    y_true = []
    y_pred = []
    
    print(f"Running inference on {len(dataset)} segments...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # Report
    print("\n" + "="*40)
    print("RESULTS ON CORRECTED SEGMENTS (Recency Sorted)")
    print("="*40)
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [CLASS_NAMES[i] for i in unique_labels]
    
    # Write to file
    with open("eval_report_final.txt", "w", encoding="utf-8") as f:
        f.write("=== Evaluation on CORRECTED Segments (300 Recent) ===\n")
        f.write(classification_report(y_true, y_pred, labels=unique_labels, target_names=[str(t) for t in target_names], digits=3))
        
        f.write("\n\nGround Truth Distribution:\n")
        from collections import Counter
        c = Counter([CLASS_NAMES[i] for i in y_true])
        for k, v in c.most_common():
            f.write(f"  {k}: {v}\n")
            
        f.write("\nPrediction Distribution:\n")
        c_pred = Counter([CLASS_NAMES[i] for i in y_pred])
        for k, v in c_pred.most_common():
            f.write(f"  {k}: {v}\n")

        sr_idx = CLASS_INDEX.get("Sinus Rhythm", 0)
        sr_preds = y_pred.count(sr_idx)
        f.write(f"\nSinus Rhythm predictions: {sr_preds}/{len(y_pred)} ({sr_preds/len(y_pred):.1%})\n")
        
        acc = accuracy_score(y_true, y_pred)
        f.write(f"\nOverall Accuracy on Corrections: {acc:.2%}\n")
        
    print("\nEvaluation report saved to eval_report_final.txt")

def collate_fn(batch):
    xs = torch.stack([torch.from_numpy(b["signal"]) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys

if __name__ == "__main__":
    evaluate()
