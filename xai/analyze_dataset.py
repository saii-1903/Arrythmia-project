#!/usr/bin/env python3
"""
analyze_dataset.py

Analyzes the class distribution in your SQL database to diagnose imbalance issues.
"""

import psycopg2
from collections import Counter
from data_loader import CLASS_NAMES, normalize_label, CLASS_INDEX

def analyze():
    conn = psycopg2.connect(
        host="localhost",
        database="ecg_analysis",
        user="ecg_user",
        password="sais"
    )
    
    with conn.cursor() as cur:
        # Get all labeled segments with raw_signal
        cur.execute("""
            SELECT arrhythmia_label, COUNT(*) as cnt
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
              AND arrhythmia_label IS NOT NULL
            GROUP BY arrhythmia_label
            ORDER BY cnt DESC
        """)
        
        raw_counts = cur.fetchall()
        
        print("=" * 70)
        print("RAW LABEL DISTRIBUTION IN DATABASE")
        print("=" * 70)
        total = sum(c[1] for c in raw_counts)
        for label, count in raw_counts:
            pct = 100 * count / total
            print(f"{label:40s} : {count:6d} ({pct:5.1f}%)")
        
        print(f"\nTotal segments: {total}")
        
        # Now check after normalization
        cur.execute("""
            SELECT arrhythmia_label
            FROM ecg_features_annotatable
            WHERE raw_signal IS NOT NULL
              AND arrhythmia_label IS NOT NULL
        """)
        
        all_labels = [row[0] for row in cur.fetchall()]
        
    conn.close()
    
    # Normalize and count
    normalized = [normalize_label(lbl) for lbl in all_labels]
    valid = [lbl for lbl in normalized if lbl in CLASS_INDEX]
    
    counts = Counter(valid)
    
    print("\n" + "=" * 70)
    print("NORMALIZED CLASS DISTRIBUTION (USED FOR TRAINING)")
    print("=" * 70)
    
    for idx, name in enumerate(CLASS_NAMES):
        count = counts.get(name, 0)
        pct = 100 * count / len(valid) if valid else 0
        print(f"{idx}: {name:40s} : {count:6d} ({pct:5.1f}%)")
    
    print(f"\nTotal usable: {len(valid)}")
    
    # Calculate imbalance ratio
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nImbalance ratio (max/min): {ratio:.1f}x")
        
        if ratio > 10:
            print("⚠️  SEVERE CLASS IMBALANCE DETECTED")
        elif ratio > 5:
            print("⚠️  Moderate class imbalance")
        else:
            print("✓ Acceptable balance")

if __name__ == "__main__":
    analyze()
