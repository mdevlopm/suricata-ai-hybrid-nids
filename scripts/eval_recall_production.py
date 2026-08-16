#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate recall/TPR of production model models/ids_model_v8_final.pkl
across attack classes (DoS, DDoS, WebAttack, Bot, Infiltration) on:
1) 20% Stratified Validation Split (seed 42)
2) Unseen / Held-out Out-of-Sample CICIDS2018 Attack Test Sets
3) Threshold sensitivity analysis (0.50, 0.66, 0.84)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from trainv8 import extract_features_v7, CLASS_MAP, INV_CLASS_MAP

MODEL_PATH = Path("/run/media/mehmet/siber data1/ai modeli xgboost/models/ids_model_v8_final.pkl")
DATA_ROOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_combined")

print("=" * 80)
print("1. LOADING PRODUCTION MODEL BUNDLE (models/ids_model_v8_final.pkl)")
print("=" * 80)
with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
threshold = float(bundle.get('threshold', 0.84))
coral = bundle.get('coral_adapter')
class_map = bundle['class_map']
inv_class_map = bundle['inv_class_map']

print(f"Model type       : {type(model).__name__}")
print(f"Active Threshold : {threshold:.4f}")
print(f"CORAL Fitted     : {coral.is_fitted_ if coral else False}")
print(f"Classes          : {class_map}")

# ============================================================================
# PART 1: 20% STRATIFIED VALIDATION SET (Exact replication of train split)
# ============================================================================
print("\n" + "=" * 80)
print("2. LOADING DATASET FOR STRATIFIED VALIDATION SPLIT")
print("=" * 80)

MAX_SAMPLES = {
    "Benign": 60_000,
    "DoS": 30_000,
    "DDoS": 30_000,
    "WebAttack": 30_000,
    "Infiltration": 30_000,
    "Bot": 30_000,
}

X_list = []
y_list = []

for cname, cidx in CLASS_MAP.items():
    fpath = DATA_ROOT / f"eve_{cname}.json"
    if not fpath.exists():
        print(f"  Warning: {fpath} not found!")
        continue
    limit = MAX_SAMPLES.get(cname, 30_000)
    feats = []
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event_type") != "flow":
                continue
            fv = extract_features_v7(ev)
            if fv is not None:
                feats.append(fv)
            if len(feats) >= limit:
                break
    arr = np.array(feats, dtype=np.float32)
    X_list.append(arr)
    y_list.append(np.full(len(feats), cidx, dtype=int))
    print(f"  Loaded {cname:<14}: {len(feats):>7,} flows (shape: {arr.shape})")

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)

# Split 80/20 with seed 42
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"\nValidation Set Size: {len(X_val):,} samples")

# Predict on Validation Set
X_val_scaled = scaler.transform(X_val)

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_val_scaled)
else:
    import xgboost as xgb
    probs = model.predict(xgb.DMatrix(X_val_scaled))

prob_attack = 1.0 - probs[:, 0]
binary_preds = (prob_attack >= threshold).astype(int)
multi_preds = probs.argmax(axis=1)

print("\n" + "=" * 80)
print(f"3. EVALUATION RESULTS ON STRATIFIED VALIDATION SET (THRESHOLD = {threshold:.2f})")
print("=" * 80)

# Calculate Recall for each attack class
results = {}
for cname, cidx in CLASS_MAP.items():
    mask = (y_val == cidx)
    n_samples = mask.sum()
    if n_samples == 0:
        continue
    if cidx == 0: # Benign -> calculate FP rate
        fp_count = (binary_preds[mask] == 1).sum()
        fpr = fp_count / n_samples
        far_str = f"{fpr*100:.2f}%"
        spec_str = f"{(1-fpr)*100:.2f}%"
        results[cname] = {
            "Total": n_samples,
            "Alarms": fp_count,
            "Rate": fpr,
            "far_str": far_str,
            "spec_str": spec_str,
        }
    else: # Attack -> calculate Recall / TPR
        tp_count = (binary_preds[mask] == 1).sum()
        recall = tp_count / n_samples
        multi_correct = (multi_preds[mask] == cidx).sum()
        multi_acc = multi_correct / n_samples
        results[cname] = {
            "Total": n_samples,
            "Detected_Binary": tp_count,
            "Recall_TPR": recall,
            "Multiclass_Acc": multi_acc,
        }

print(f"{'Class':<15} | {'Total Samples':>13} | {'Detected':>10} | {'Recall / TPR (@ ' + str(threshold) + ')':>26} | {'Multi-Class Recall':>18}")
print("-" * 90)
for cname, cidx in CLASS_MAP.items():
    if cname == "Benign":
        res = results[cname]
        print(f"{cname:<15} | {res['Total']:>13,} | {res['Alarms']:>10,} | {'FAR: ' + res['far_str']:>26} | {'Specificity: ' + res['spec_str']:>18}")
    else:
        res = results[cname]
        rec_str = f"{res['Recall_TPR']*100:.2f}%"
        multi_str = f"{res['Multiclass_Acc']*100:.2f}%"
        print(f"{cname:<15} | {res['Total']:>13,} | {res['Detected_Binary']:>10,} | {rec_str:>26} | {multi_str:>18}")

# Overall Attack Recall
attack_mask = (y_val > 0)
total_attacks = attack_mask.sum()
detected_attacks = (binary_preds[attack_mask] == 1).sum()
overall_recall = detected_attacks / total_attacks

print("-" * 90)
print(f"{'OVERALL ATTACK':<15} | {total_attacks:>13,} | {detected_attacks:>10,} | {f'{overall_recall*100:.2f}%':>26} |")
print("=" * 90)

# ============================================================================
# PART 2: HELD-OUT / UNSEEN CICIDS2018 ATTACK SAMPLES (Out-of-sample slice)
# ============================================================================
print("\n" + "=" * 80)
print("4. TESTING ON COMPLETELY UNSEEN HELD-OUT SLICES (Lines > 150,000)")
print("=" * 80)

held_out_results = {}
for cname in ["DoS", "DDoS", "WebAttack"]:
    fpath = DATA_ROOT / f"eve_{cname}.json"
    feats = []
    # Skip first 150,000 lines (which were training pool) and take 20,000 completely fresh lines
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            if idx < 150_000:
                continue
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event_type") != "flow":
                continue
            fv = extract_features_v7(ev)
            if fv is not None:
                feats.append(fv)
            if len(feats) >= 20_000:
                break
    
    X_ho = np.array(feats, dtype=np.float32)
    X_ho_s = scaler.transform(X_ho)
    if hasattr(model, "predict_proba"):
        p_ho = model.predict_proba(X_ho_s)
    else:
        import xgboost as xgb
        p_ho = model.predict(xgb.DMatrix(X_ho_s))
    
    p_atk = 1.0 - p_ho[:, 0]
    preds_bin = (p_atk >= threshold).astype(int)
    rec = preds_bin.mean()
    held_out_results[cname] = {
        "Total": len(feats),
        "Detected": int(preds_bin.sum()),
        "Recall": rec
    }
    print(f"  Unseen {cname:<12} (20k flows from slice 150k+): Detected {preds_bin.sum():,}/{len(feats):,} -> Recall: {rec*100:.2f}%")

print("=" * 80)

# ============================================================================
# PART 3: THRESHOLD SENSITIVITY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("5. THRESHOLD SENSITIVITY ON ATTACK RECALL (Validation Set)")
print("=" * 80)
print(f"{'Threshold':<10} | {'DoS':>10} | {'DDoS':>10} | {'WebAttack':>10} | {'Infiltration':>12} | {'Bot':>10} | {'Overall Atk':>12} | {'FAR (Benign)':>12}")
print("-" * 90)

for thr in [0.50, 0.60, 0.66, 0.70, 0.75, 0.80, 0.84, 0.88, 0.90]:
    b_preds = (prob_attack >= thr).astype(int)
    row = []
    for cname, cidx in CLASS_MAP.items():
        mask = (y_val == cidx)
        if cidx == 0:
            far = (b_preds[mask] == 1).mean()
        else:
            rec = (b_preds[mask] == 1).mean()
            row.append(f"{rec*100:.2f}%")
    
    all_atk_rec = (b_preds[attack_mask] == 1).mean()
    print(f"{thr:<10.2f} | {row[0]:>10} | {row[1]:>10} | {row[2]:>10} | {row[3]:>12} | {row[4]:>10} | {all_atk_rec*100:>11.2f}% | {far*100:>11.2f}%")

print("=" * 90)
