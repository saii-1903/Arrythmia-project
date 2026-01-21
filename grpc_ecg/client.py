import sys
from pathlib import Path
import time
import grpc
import math
import random

# Add proto directory to sys.path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "proto"))

import ecg_pb2
import ecg_pb2_grpc

def generate_ecg_data(sample_rate=125):
    """Generates synthetic ECG data stream at a specific sample rate."""
    device_id = "DEV-001"
    fs = sample_rate
    t = 0
    
    print(f"Simulating ECG stream @ {fs}Hz...")
    
    # Simulate a stream indefinitely
    while True:
        # Send chunks of data. For 125Hz, let's send 12 or 13 samples every 0.1s.
        chunk_size = int(fs * 0.1) 
        values = []
        for _ in range(chunk_size):
            # Synthetic signal: 1Hz sine wave + noise
            val = math.sin(2 * math.pi * 1.0 * t / fs) + (random.random() * 0.1)
            values.append(val)
            t += 1
            
        yield ecg_pb2.ECGData(
            values=values,
            device_id=device_id,
            timestamp=int(time.time() * 1000),
            sample_rate=fs
        )
        
        time.sleep(0.1) # Simulate real-time delay

def run():
    print("Connecting to gRPC server...")
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = ecg_pb2_grpc.ECGServiceStub(channel)
        
        try:
            print("Starting stream...")
            # We can test with 125Hz here to see if the server resamples correctly
            responses = stub.StreamECG(generate_ecg_data(sample_rate=125))
            
            for alert in responses:
                print(f"\n[ALERT RECEIVED]")
                print(f"  Type: {alert.arrhythmia_type}")
                print(f"  Conf: {alert.confidence:.4f}")
                print(f"  Msg:  {alert.message}")
                print("-" * 30)
                
        except grpc.RpcError as e:
            print(f"gRPC Error: {e.code()} - {e.details()}")
        except KeyboardInterrupt:
            print("Stream stopped by user.")

if __name__ == '__main__':
    run()
