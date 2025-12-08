#!/usr/bin/env python3
"""
predict_and_explain.py

Usage:
  python src/predict_and_explain.py \
    --ckpt outputs/checkpoints/best_epoch_8.pt \
    --data data/merged_jsons \
    --out predictions.csv \
    --plots out/explain_plots \
    --n 200

Notes:
 - Expects model architecture in src/models.py (CNNTransformerClassifier).
 - Expects dataset loader in src/data_loader.py (ECGRawDataset).
 - Produces per-sample CSV with predicted class, probabilities, and features.
 - Produces per-sample PNGs showing ECG with saliency overlays for top predicted class.
"""

import os
import argparse
from pathlib import Path
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, find_peaks

# Import project modules (assumes working dir has src/ or is project root)
from models import CNNTransformerClassifier
from data_loader import ECGRawDataset, CLASS_NAMES, TARGET_FS, SEGMENT_LEN

# -----------------------
# Feature extraction
# -----------------------
def bandpass_filter(sig, fs, low=0.5, high=45.0, order=3):
    ny = 0.5 * fs
    lowb, highb = low / ny, high / ny
    b, a = butter(order, [lowb, highb], btype="band")
    return filtfilt(b, a, sig)

def estimate_hr_from_peaks(sig, fs):
    """
    Very simple R-peak based HR estimate:
    - bandpass 5-25 Hz (helps R-peaks)
    - find peaks with prominence and distance
    - return median heart-rate (bpm)
    """
    try:
        s = bandpass_filter(sig, fs, low=5.0, high=25.0, order=3)
        # absolute and smoothing
        abs_s = np.abs(s)
        # find peaks: enforce min distance ~0.3s and prominence
        min_dist = int(0.3 * fs)
        peaks, _ = find_peaks(abs_s, distance=min_dist, prominence=(np.std(abs_s) * 0.5))
        if len(peaks) < 2:
            return None, peaks
        rr = np.diff(peaks) / float(fs)  # seconds
        hr = 60.0 / np.median(rr)
        return float(hr), peaks
    except Exception:
        return None, []

def spectral_band_powers(sig, fs):
    """Return spectral power in coarse bands useful for ECG (0-0.5,0.5-3,3-8,8-40 Hz)."""
    f, Pxx = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
    bands = [(0.0, 0.5), (0.5, 3.0), (3.0, 8.0), (8.0, 40.0)]
    powers = []
    total = np.trapz(Pxx, f) + 1e-12
    for a, b in bands:
        mask = (f >= a) & (f < b)
        p = np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0
        powers.append(float(p / total))
    # dominant frequency
    domf = float(f[np.argmax(Pxx)])
    return powers, domf

def extract_features(sig, fs):
    """
    Returns a dict of extracted features:
      - mean, std, min, max, ptp
      - rms
      - skewness, kurtosis
      - band powers (4 bands) normalized
      - dominant frequency
      - hr_est (bpm) if found
    """
    from scipy.stats import skew, kurtosis
    sig = np.asarray(sig, dtype=float)
    feats = {}
    feats["mean"] = float(np.mean(sig))
    feats["std"] = float(np.std(sig))
    feats["min"] = float(np.min(sig))
    feats["max"] = float(np.max(sig))
    feats["ptp"] = float(np.ptp(sig))
    feats["rms"] = float(np.sqrt(np.mean(sig**2)))
    feats["skew"] = float(skew(sig))
    feats["kurtosis"] = float(kurtosis(sig))
    band_powers, domf = spectral_band_powers(sig, fs)
    for i, p in enumerate(band_powers):
        feats[f"band_power_{i}"] = float(p)
    feats["dom_freq"] = domf
    hr, peaks = estimate_hr_from_peaks(sig, fs)
    feats["hr_est"] = float(hr) if hr is not None else None
    feats["n_peaks"] = int(len(peaks))
    return feats, peaks

# -----------------------
# Saliency explanation
# -----------------------
def compute_input_gradients(model, x_tensor, target_class_idx):
    """
    Compute gradient of target logit wrt input signal.
    x_tensor: shape (1, L) float32 requires grad False -> we will create a copy that requires grad
    Returns gradient numpy array shape (L,)
    """
    model.eval()
    x = x_tensor.clone().detach().float().unsqueeze(0)  # (1, L)
    x = x.to(next(model.parameters()).device)
    x.requires_grad_()
    logits = model(x)  # (1, C)
    # take logit for target class
    target_logit = logits[0, target_class_idx]
    # backward
    model.zero_grad()
    target_logit.backward(retain_graph=False)
    grad = x.grad.detach().cpu().numpy()[0]  # (L,)
    return grad

# -----------------------
# Plotting helpers
# -----------------------
def plot_signal_with_saliency(sig, fs, saliency, peaks, outpath, title=None):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(sig)) / float(fs)
    # normalize saliency for overlay
    sabs = np.abs(saliency)
    if sabs.max() > 0:
        sal_plot = sabs / sabs.max() * (0.35 * (np.max(sig) - np.min(sig)))
    else:
        sal_plot = sabs
    plt.figure(figsize=(12,3))
    plt.plot(t, sig, color='k', linewidth=0.8, label='ECG')
    plt.fill_between(t, np.min(sig), np.min(sig) + sal_plot, color='red', alpha=0.35, label='saliency (abs)')
    if peaks is not None and len(peaks)>0:
        plt.scatter(np.array(peaks)/fs, sig[peaks], color='blue', s=10, label='R-peaks')
    plt.xlabel("Time (s)")
    plt.title(title or "")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# -----------------------
# Main inference pipeline
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (pt file)")
    ap.add_argument("--data", required=True, help="Path to merged JSON folder (or data root)")
    ap.add_argument("--out", required=True, help="CSV output file path")
    ap.add_argument("--plots", required=True, help="Directory to save explain plots")
    ap.add_argument("--n", type=int, default=None, help="Max number of samples to process")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for inference")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    data_root = Path(args.data)
    out_csv = Path(args.out)
    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model (architecture must match training)
    num_classes = len(CLASS_NAMES)
    model = CNNTransformerClassifier(num_classes=num_classes)
    ck = torch.load(str(ckpt_path), map_location=device)
    # ck may contain "model_state" or state_dict directly
    if "model_state" in ck:
        model.load_state_dict(ck["model_state"])
    elif "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    else:
        model.load_state_dict(ck)
    model.to(device)
    model.eval()

    # Load data (use data_loader.ECGRawDataset)
    dataset = ECGRawDataset(json_dir=None, from_sql=False, data_roots=[data_root])
    n_total = len(dataset)
    print("Loaded dataset size:", n_total)
    limit = args.n if args.n is not None else n_total

    # Prepare CSV header
    feats_keys_example, _ = extract_features(np.zeros(SEGMENT_LEN), TARGET_FS)
    feat_names = list(feats_keys_example.keys())
    header = ["file", "pred_class_idx", "pred_class", "pred_prob"] + [f"prob_class_{i}" for i in range(num_classes)] + feat_names

    # Open CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)

        processed = 0
        for idx in range(n_total):
            if processed >= limit:
                break
            item = dataset[idx]
            sig = item["signal"]  # np array length SEGMENT_LEN
            label_int = int(item["label"]) if "label" in item else None
            meta = item.get("meta", {})
            source = meta.get("_source_path") if isinstance(meta, dict) and meta.get("_source_path") else item.get("meta", {}).get("source") or item.get("_source_path", "unknown")
            # Prepare input tensor (model expects shape (B,L) and will unsqueeze)
            x = torch.from_numpy(sig).float().to(device).unsqueeze(0)  # (1,L)
            with torch.no_grad():
                logits = model(x)  # (1,C)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0].tolist()
                pred_idx = int(np.argmax(probs))
                pred_prob = float(probs[pred_idx])

            # Extract features
            feats, peaks = extract_features(sig, TARGET_FS)

            # Compute saliency for predicted class (input-grad)
            try:
                grad = compute_input_gradients(model, torch.from_numpy(sig).float(), pred_idx)
                # direction signature: positive gradient means increasing input increases logit
                # We'll save absolute saliency and allow inspection of sign via sign(grad)
            except Exception as ex:
                print("Saliency failed for idx", idx, ":", ex)
                grad = np.zeros_like(sig)
                peaks = []

            # Save plot for a few samples (top-k or all)
            plot_path = plots_dir / f"{idx:05d}__pred{pred_idx}_{CLASS_NAMES[pred_idx].replace(' ','_')}.png"
            title = f"Pred: {CLASS_NAMES[pred_idx]} ({pred_prob:.2f})  True:{CLASS_NAMES[label_int] if label_int is not None else 'NA'}"
            plot_signal_with_saliency(sig, TARGET_FS, grad, peaks, str(plot_path), title=title)

            # write row
            row = [str(source), pred_idx, CLASS_NAMES[pred_idx], pred_prob] + probs + [feats[k] for k in feat_names]
            writer.writerow(row)

            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed}/{limit}")

    print("Done. CSV saved to:", out_csv)
    print("Plots saved to:", plots_dir)

if __name__ == "__main__":
    main()
