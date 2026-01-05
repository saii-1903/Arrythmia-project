# Model Architecture Definition

**Status**: FROZEN
**Architecture**: Hybrid CNN-Transformer

## 1. Architectural Intent
The project enforces a specific division of labor between the two main components of the neural network. This choice is deliberate to address the dual nature of ECG analysis: **Morphology** (Shape) and **Rhythm** (Time).

### A. CNN (Convolutional Neural Network)
*   **Component**: `SmallCNN` (in `models.py`)
*   **Role**: **Morphological Feature Extractor**
*   **Why**: Arrythmias like Bundle Branch Blocks or PVCs are defined by the *shape* of individual heartbeats (wide QRS, notched R-wave).
*   **Function**: Convolves over small, local windows of the raw signal to create dense embeddings representing the shape of local waveform segments.

### B. Transformer (Encoder)
*   **Component**: `TransformerEncoder` (in `models.py`)
*   **Role**: **Temporal Sequence Modeler**
*   **Why**: Arrythmias like Atrial Fibrillation or Bigeminy are defined by the *timing and pattern* of beats over time (irregularity, repeating patterns).
*   **Function**: Uses Self-Attention to look across the entire 10-second sequence of CNN embeddings. It learns dependencies like "Every second beat is different" (Bigeminy) or "No pattern exists" (AFib).

## 2. Rationale for Hybrid Approach
We rejected a "Pure CNN" approach because it struggles to link events 5 seconds apart (long-range context).
We rejected a "Pure RNN/LSTM" approach due to training instability and slower inference.
The **CNN-Transformer** combines the best of both:
*   Standard (CNN) efficiency for shape.
*   State-of-the-art (Transformer) attention for rhythm.

---
*This document serves as the ground truth for the model's design philosophy. Any deviations (e.g., removing the Transformer) require a formal architectural review.*
