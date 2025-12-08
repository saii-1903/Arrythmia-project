import torch
from pathlib import Path
from data_loader import CLASS_NAMES

ckpt_path = Path("outputs/checkpoints/best_model.pth")

print(f"Checking {ckpt_path}...")
if not ckpt_path.exists():
    print("❌ File does not exist!")
    exit(1)

try:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("✅ File loaded successfully")
    
    if "model_state" in state:
        sd = state["model_state"]
        print("✅ Found 'model_state' key")
    else:
        sd = state
        print("⚠️ 'model_state' key not found, assuming direct state dict")
        
    print(f"Keys in state dict: {len(sd)}")
    
    # Check output layer size
    # Usually the last layer is 'fc.weight' or similar
    last_key = list(sd.keys())[-1]
    print(f"Last key: {last_key}")
    
    if "fc.weight" in sd:
        shape = sd["fc.weight"].shape
        print(f"fc.weight shape: {shape}")
        print(f"Expected output size: {len(CLASS_NAMES)}")
        
        if shape[0] != len(CLASS_NAMES):
            print(f"❌ Mismatch! Model has {shape[0]} outputs, but CLASS_NAMES has {len(CLASS_NAMES)}")
        else:
            print("✅ Output size matches CLASS_NAMES")
            
    if "class_names" in state:
        saved_classes = state["class_names"]
        print(f"Saved class names: {saved_classes}")
        if saved_classes != CLASS_NAMES:
            print("❌ Saved class names do not match current CLASS_NAMES")
            print(f"Current: {CLASS_NAMES}")
        else:
            print("✅ Class names match")
            
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Error loading checkpoint: {e}")
