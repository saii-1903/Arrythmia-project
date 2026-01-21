import sys
from pathlib import Path
import time
import grpc
from concurrent import futures
import torch
import numpy as np
from scipy.signal import resample

# Add project root and proto directory to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir / "proto"))

# Now we can import project modules and generated proto code
from models_training.models import CNNTransformerClassifier
from models_training.data_loader import CLASS_NAMES
import ecg_pb2
import ecg_pb2_grpc

# Constants
TARGET_FS = 250
WINDOW_SECONDS = 10
WINDOW_SAMPLES_TARGET = TARGET_FS * WINDOW_SECONDS # 2500
SLIDE_SECONDS = 1

class ECGStreamServicer(ecg_pb2_grpc.ECGServiceServicer):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def StreamECG(self, request_iterator, context):
        print("New connection established.")
        buffer = []
        current_fs = 250 # Default heartbeat
        
        for packet in request_iterator:
            # packet is ECGData
            if packet.values:
                # Update fs if provided (though it should be constant per stream)
                if packet.sample_rate > 0:
                    current_fs = packet.sample_rate
                
                buffer.extend(packet.values)
                
                # Window size in samples depends on current_fs
                window_samples_source = current_fs * WINDOW_SECONDS
                slide_samples_source = current_fs * SLIDE_SECONDS
                
                # Check if we have enough data for a window
                while len(buffer) >= window_samples_source:
                    # Extract window
                    window_data = np.array(buffer[:window_samples_source], dtype=np.float32)
                    
                    # RESAMPLING LOGIC
                    if current_fs != TARGET_FS:
                        # Resample to 2500 samples
                        window_data = resample(window_data, WINDOW_SAMPLES_TARGET).astype(np.float32)
                    
                    # Prepare for model
                    # Model expects (B, 1, L) or (B, L) depending on implementation
                    # models.py says: if x.dim() == 2: x = x.unsqueeze(1)
                    tensor_data = torch.from_numpy(window_data).unsqueeze(0).float().to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        logits = self.model(tensor_data)
                        probs = torch.softmax(logits, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_idx].item()
                    
                    class_name = CLASS_NAMES[pred_idx]
                    
                    # Alert logic
                    # 0: Sinus Rhythm, 21: Artifact
                    if pred_idx != 0 and pred_idx != 21:
                         print(f"ALERT: {class_name} detected! Conf: {confidence:.2f} (Input FS: {current_fs})")
                         yield ecg_pb2.ArrhythmiaAlert(
                             arrhythmia_type=class_name,
                             confidence=confidence,
                             message=f"Detected {class_name} with {confidence:.1%} confidence. Analyzed @{current_fs}Hz resampled to 250Hz",
                             timestamp=int(time.time() * 1000)
                         )
                    
                    # Slide window
                    buffer = buffer[slide_samples_source:]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = CNNTransformerClassifier(num_classes=len(CLASS_NAMES))
    ckpt_path = project_root / "models_training" / "outputs" / "checkpoints" / "best_model.pth"
    
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Using random weights.")
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
            
    model.to(device)
    model.eval()
    return model, device

def serve():
    model, device = load_model()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ecg_pb2_grpc.add_ECGServiceServicer_to_server(
        ECGStreamServicer(model, device), server
    )
    
    port = 50051
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC ECG Server started on port {port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Server stopped")

if __name__ == '__main__':
    serve()
