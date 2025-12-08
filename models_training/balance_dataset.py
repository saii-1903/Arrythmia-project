#!/usr/bin/env python3
"""
balance_dataset.py

Helps balance your dataset by:
1. Identifying which classes need more samples
2. Suggesting how many samples to add
3. Optionally creating synthetic samples using SMOTE-like technique
"""

import psycopg2
import numpy as np
from collections import Counter
from data_loader import CLASS_NAMES, normalize_label, CLASS_INDEX

def analyze_and_suggest():
    conn = psycopg2.connect(
        host="localhost",
        database="ecg_analysis",
        user="ecg_user",
        password="sais"
    )
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT arrhythmia_label, raw_signal
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
              AND arrhythmia_label IS NOT NULL
        """)
        
        rows = cur.fetchall()
    
    conn.close()
    
    # Normalize and count
    class_samples = {name: [] for name in CLASS_NAMES}
    
    for label, signal in rows:
        norm_label = normalize_label(label)
        if norm_label in CLASS_INDEX:
            class_samples[norm_label].append(np.array(signal))
    
    counts = {name: len(samples) for name, samples in class_samples.items()}
    
    print("=" * 70)
    print("CURRENT CLASS DISTRIBUTION")
    print("=" * 70)
    
    total = sum(counts.values())
    for idx, name in enumerate(CLASS_NAMES):
        count = counts[name]
        pct = 100 * count / total if total > 0 else 0
        print(f"{idx}: {name:40s} : {count:6d} ({pct:5.1f}%)")
    
    # Find target count (median or mean of top 3 classes)
    sorted_counts = sorted(counts.values(), reverse=True)
    if len(sorted_counts) >= 3:
        target = int(np.median(sorted_counts[:3]))
    else:
        target = max(counts.values()) // 2
    
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED TARGET: {target} samples per class")
    print(f"{'=' * 70}\n")
    
    print("Samples needed per class:")
    for idx, name in enumerate(CLASS_NAMES):
        current = counts[name]
        needed = max(0, target - current)
        if needed > 0:
            print(f"{idx}: {name:40s} : Need {needed:6d} more samples")
        else:
            print(f"{idx}: {name:40s} : ✓ Sufficient ({current} samples)")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    # Check if any class has < 100 samples
    critical_classes = [name for name, count in counts.items() if count < 100]
    
    if critical_classes:
        print("\n⚠️  CRITICAL: These classes have < 100 samples:")
        for name in critical_classes:
            print(f"   - {name}: {counts[name]} samples")
        print("\nACTIONS:")
        print("1. Find and import more real data for these classes")
        print("2. Check if your label normalization is too aggressive")
        print("3. Consider merging similar classes (e.g., all Sinus variants)")
        print("4. Use data augmentation (already in train_balanced.py)")
    
    # Check overall imbalance
    if counts:
        max_count = max(counts.values())
        min_count = min([c for c in counts.values() if c > 0])
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nCurrent imbalance ratio: {ratio:.1f}x")
        
        if ratio > 50:
            print("⚠️  SEVERE imbalance - Model will struggle")
            print("\nSTRONGLY RECOMMENDED:")
            print("1. Use train_balanced.py (Focal Loss + aggressive sampling)")
            print("2. Add more minority class samples to database")
            print("3. Consider rule-based classification for rare classes")
        elif ratio > 10:
            print("⚠️  Moderate imbalance")
            print("RECOMMENDED: Use train_balanced.py")
        else:
            print("✓ Acceptable balance - regular train.py should work")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run: python train_balanced.py")
    print("   (Uses Focal Loss + aggressive oversampling)")
    print("")
    print("2. If results still poor, consider:")
    print("   - Importing more data for minority classes")
    print("   - Using rule-based classification for rare arrhythmias")
    print("   - Combining similar classes to reduce imbalance")

if __name__ == "__main__":
    analyze_and_suggest()
