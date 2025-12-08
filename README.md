# ECG Arrhythmia AI Workstation

A comprehensive system for generating, training, visualizing, and explaining ECG arrhythmias using sophisticated Transformer-based AI.

## 📂 Project Structure

## 📂 Project Structure

### `dashboard/`
*   **`app.py`**: The main Web Dashboard (Flask). Run this to start the UI.
*   **`templates/` & `static/`**: Frontend assets.

### `simulation/`
*   **`ECG_Workstation.py`**: Desktop GUI for real-time ECG simulation.
*   **`synthesize_arrhythmias.py`**: Generates synthetic arrhythmia data.

### `models_training/`
*   **`train_balanced.py`**: Robust training script (Focal Loss, Resampling).
*   **`data_loader.py`**: Dataset and Class definitions.
*   **`models.py`**: Neural Network architecture.
*   **`outputs/`**: Trained checkpoints (`best_model.pth`).

### `xai/`
*   **`xai.py`**: Explainable AI engine (Inference + Clinical Rules).

### `database/`
*   **`db_service.py`**: Database interaction logic.
*   **`export_corrected_segments`**: Tools to export SQL data for training.

### `utils/`
*   **`ecgprocessor.py`**: Signal processing helpers.

### `data/`
*   Shared folder for `ecg_data` (uploads) and `input_segments` (bulk data).

---

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Web Dashboard
```bash
cd dashboard
python app.py
```
*   **URL**: `http://localhost:5000`

### 3. Run the Desktop Simulator
```bash
cd simulation
python ECG_Workstation.py
```

### 4. Retraining
Use the dashboard button or run manually:
```bash
cd models_training
python train_balanced.py
```

---

## ⚕️ Supported Arrhythmias
The system supports **31+ Classes**, including:
*   **Baseline**: Sinus (Normal, Brady, Tachy), AFib, Flutter, Junctional, VT, VFib, Heart Blocks (1st, 2nd I/II, 3rd).
*   **Ectopic**: PACs, PVCs (Singles, Couplets, Triplets).
*   **Patterns**: Bigeminy, Trigeminy, Quadrigeminy.
*   **Combinations**: e.g., "Sinus Tachycardia + PVC".
