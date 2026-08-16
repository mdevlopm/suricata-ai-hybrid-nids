#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Comparison Script:
Compares:
1) Current Prod (v8_final) - Direct Scaler (Threshold 0.84)
2) Current Prod (v8_final) - With Target CORAL (Threshold 0.84)
3) Reference v7 (v7_final) - Direct Scaler (Threshold 0.66)
4) Previous Buggy v8 (v8_control) - Direct / Old CORAL (Threshold 0.67)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split

from trainv8 import extract_features_v7, CLASS_MAP

DATA_ROOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_combined")

# Load 20k per class for fast, exact validation split
X_list, y_list = [], []
for cname, cidx in CLASS_MAP.items():
    fpath = DATA_ROOT / f"eve_{cname}.json"
    feats = []
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip(): continue
            try: ev = json.loads(line)
            except Exception: continue
            if ev.get("event_type") != "flow": continue
            fv = extract_features_v7(ev)
            if fv is not None: feats.append(fv)
            if len(feats) >= 20_000: break
    X_list.append(np.array(feats, dtype=np.float32))
    y_list.append(np.full(len(feats), cidx, dtype=int))

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Models to evaluate
models_info = [
    ("v8_final (Current Live, No CORAL)", "models/ids_model_v8_final.pkl", False, 0.84),
    ("v8_final (Current Live, WITH CORAL)", "models/ids_model_v8_final.pkl", True, 0.84),
    ("v7_final (Stable Reference)", "archive/models/ids_model_v7_final.pkl", False, 0.66),
    ("v8_control (Pre-Deploy Buggy Backup)", "models/ids_model_v8_control_final.pkl", False, 0.67),
]

print("=" * 110)
print("COMPREHENSIVE MULTI-MODEL RECALL & FAR EVALUATION ON CICIDS2018 VALIDATION SET")
print("=" * 110)
print(f"{'Model Configuration':<38} | {'DoS':>8} | {'DDoS':>8} | {'WebAtk':>8} | {'Bot':>8} | {'Infiltr':>8} | {'All Attack':>10} | {'FAR (Benign)':>12}")
print("-" * 110)

for mlabel, mpath, use_coral, thr in models_info:
    p = Path(mpath)
    if not p.exists():
        print(f"{mlabel:<38} | File not found: {mpath}")
        continue
    with open(p, 'rb') as f:
        bundle = pickle.load(f)
    m = bundle['model']
    sc = bundle['scaler']
    cor = bundle.get('coral_adapter')
    
    # Process features
    X_eval = X_val.copy()
    if use_coral and cor is not None:
        X_eval = cor.transform_target(X_eval)
    
    X_s = sc.transform(X_eval)
    
    if hasattr(m, 'predict_proba'):
        p_eval = m.predict_proba(X_s)
    else:
        import xgboost as xgb
        p_eval = m.predict(xgb.DMatrix(X_s))
    
    p_atk = 1.0 - p_eval[:, 0]
    preds = (p_atk >= thr).astype(int)
    
    rec_dos = (preds[y_val == 1] == 1).mean() * 100
    rec_ddos = (preds[y_val == 2] == 1).mean() * 100
    rec_web = (preds[y_val == 3] == 1).mean() * 100
    rec_infil = (preds[y_val == 4] == 1).mean() * 100
    rec_bot = (preds[y_val == 5] == 1).mean() * 100
    rec_all = (preds[y_val > 0] == 1).mean() * 100
    far_ben = (preds[y_val == 0] == 1).mean() * 100
    
    print(f"{mlabel:<38} | {rec_dos:>7.2f}% | {rec_ddos:>7.2f}% | {rec_web:>7.2f}% | {rec_bot:>7.2f}% | {rec_infil:>7.2f}% | {rec_all:>9.2f}% | {far_ben:>11.2f}%")

print("=" * 110)
