
import os
import shutil
import subprocess
from pathlib import Path
import sys

# Paths
BASE_DIR = Path(__file__).resolve().parent
RETRAIN_SCRIPT = BASE_DIR / "models_training" / "retrain.py"
VALIDATION_SCRIPT = BASE_DIR / "validation" / "automated_test.py"
OUTPUT_DIR = BASE_DIR / "frozen_release"

def main():
    print(">>> STARTING RELEASE PIPELINE")
    print("="*60)
    
    # 1. Start Training
    epochs = 3 # Fast Run for validation
    print(f"\n[Step 1] Training Model (Epochs={epochs})...")
    
    try:
        cmd = [sys.executable, str(RETRAIN_SCRIPT), "--epochs", str(epochs)]
        subprocess.check_call(cmd)
        print("[+] Training Complete.")
    except subprocess.CalledProcessError as e:
        print(f"[X] Training Failed: {e}")
        return

    # 2. Validation
    print(f"\n[Step 2] Validating Pipeline...")
    try:
        # Run 5 samples per class
        cmd = [sys.executable, str(VALIDATION_SCRIPT)]
        # Use python -c to run main? No, just run the script.
        # But we need to pass args? automated_test currently uses default 5.
        subprocess.check_call(cmd)
        print("[+] Validation Suite Passed.")
    except subprocess.CalledProcessError as e:
        print(f"[X] Validation Failed: {e}")
        # We continue to freeze? Usually no.
        # But for this demo, let's allow "freezing the artifacts we have" or stop.
        # Let's stop.
        print("Stopping release due to validation failure.")
        return

    # 3. Freeze Model
    print(f"\n[Step 3] Freezing Model & Artifacts...")
    
    # Create folder
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    
    # Copy items
    artifacts = [
        BASE_DIR / "models_training/outputs/checkpoints/best_model.pth",
        BASE_DIR / "signal_processing/config.yaml",
        BASE_DIR / "models_training/MODEL_INTENT.md",
        BASE_DIR / "models_training/metrics.py",
        BASE_DIR / "validation/history.csv"
    ]
    
    for src in artifacts:
        if src.exists():
            dst = OUTPUT_DIR / src.name
            shutil.copy2(src, dst)
            print(f"  -> Copied {src.name}")
        else:
            print(f"  [!] Warning: Artifact not found {src}")
            
    print(f"\n[+] RELEASE COMPLETE. Artifacts stored in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
