
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import from local modules
# data_loader and models are in models_training, so we can import directly if we are in that dir,
# but since we appended project root, we should be careful.
# retrain.py uses:
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from data_loader import ...
# meaning data_loader is expected to be Importable from sys.path or local.
# Wait, data_loader is in models_training.
# If I run this script from models_training, I can import data_loader directly.

sys.path.append(str(BASE_DIR / "models_training"))

from data_loader import ECGRawDatasetSQL, CLASS_NAMES
from models import CNNTransformerClassifier

def evaluate():
    print("=== Model Evaluation on SQL Data ===")
    
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Dataset
    print("Loading dataset from SQL...")
    dataset = ECGRawDatasetSQL()
    print(f"Dataset size: {len(dataset)} segments")
    
    if len(dataset) == 0:
        print("No data found in SQL database.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 3. Load Model
    ckpt_path = BASE_DIR / "models_training" / "outputs" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading model from {ckpt_path}...")
    model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(ckpt_path, map_location=device)
    
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
        
    model.to(device)
    model.eval()

    # 4. Run Inference
    y_true = []
    y_pred = []
    
    print("Running inference...")
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    # 5. Report
    from metrics import print_detailed_report
    
    # Filter class names to only those present in the evaluation set
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = [CLASS_NAMES[i] for i in unique_labels]
    
    # Check if target_names is empty (edge case: empty DB)
    if not target_names:
        print("No data to evaluate.")
        return

    # Use the formal clinical reporting module
    print_detailed_report(y_true, y_pred, target_names)

def collate_fn(batch):
    # Reusing the simple collate from retrain.py logic
    xs = torch.stack([torch.from_numpy(b["signal"]) for b in batch], dim=0).float()
    ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return xs, ys

if __name__ == "__main__":
    evaluate()
