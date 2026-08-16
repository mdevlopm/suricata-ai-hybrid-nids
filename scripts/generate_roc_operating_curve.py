#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate fine-grained Operating Curve (ROC / PR Sweep) for models/ids_model_v8_final.pkl
Sweeping threshold from 0.50 to 0.99 in 0.01 steps.
Outputs the full table of:
Threshold | Thursday Benign FAR (%) | Thursday FP Count | DoS TPR | DDoS TPR | WebAtk TPR | Bot TPR | Infil TPR | Overall Attack TPR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split

from trainv8 import extract_features_v7, CLASS_MAP

MODEL_PATH = Path("/run/media/mehmet/siber data1/ai modeli xgboost/models/ids_model_v8_final.pkl")
DATA_ROOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_combined")
THURSDAY_EVE = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/eve/cicids2017_thursday_eve.json")

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']

# 1. Load Thursday Benign Traffic
thurs_feats = []
with open(THURSDAY_EVE, 'r') as f:
    for line in f:
        if not line.strip(): continue
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("event_type") != "flow": continue
        fv = extract_features_v7(ev)
        if fv is not None: thurs_feats.append(fv)

X_thurs = np.array(thurs_feats, dtype=np.float32)
X_thurs_s = scaler.transform(X_thurs)

if hasattr(model, 'predict_proba'):
    p_thurs = model.predict_proba(X_thurs_s)
else:
    import xgboost as xgb
    p_thurs = model.predict(xgb.DMatrix(X_thurs_s))

p_thurs_atk = 1.0 - p_thurs[:, 0]
n_thurs = len(X_thurs)

# 2. Load Attack Validation Dataset (CICIDS2018)
X_list, y_list = [], []
MAX_SAMPLES = {
    "Benign": 50_000,
    "DoS": 30_000,
    "DDoS": 30_000,
    "WebAttack": 30_000,
    "Infiltration": 30_000,
    "Bot": 30_000,
}

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
            if len(feats) >= MAX_SAMPLES.get(cname, 30_000): break
    X_list.append(np.array(feats, dtype=np.float32))
    y_list.append(np.full(len(feats), cidx, dtype=int))

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

X_val_s = scaler.transform(X_val)

if hasattr(model, 'predict_proba'):
    p_val = model.predict_proba(X_val_s)
else:
    import xgboost as xgb
    p_val = model.predict(xgb.DMatrix(X_val_s))

p_val_atk = 1.0 - p_val[:, 0]

# Precompute masks
mask_dos = (y_val == 1)
mask_ddos = (y_val == 2)
mask_web = (y_val == 3)
mask_infil = (y_val == 4)
mask_bot = (y_val == 5)
mask_all_atk = (y_val > 0)

thresholds = np.arange(0.50, 1.00, 0.01)

results_table = []

for thr in thresholds:
    # Thursday FAR & FP count
    fp_thurs = (p_thurs_atk >= thr).sum()
    far_thurs = (fp_thurs / n_thurs) * 100.0
    
    # Attack TPR
    preds_val = (p_val_atk >= thr).astype(int)
    tpr_dos = (preds_val[mask_dos] == 1).mean() * 100.0
    tpr_ddos = (preds_val[mask_ddos] == 1).mean() * 100.0
    tpr_web = (preds_val[mask_web] == 1).mean() * 100.0
    tpr_bot = (preds_val[mask_bot] == 1).mean() * 100.0
    tpr_infil = (preds_val[mask_infil] == 1).mean() * 100.0
    tpr_all = (preds_val[mask_all_atk] == 1).mean() * 100.0
    
    results_table.append({
        'threshold': float(thr),
        'far_thurs': far_thurs,
        'fp_count': int(fp_thurs),
        'n_thurs': n_thurs,
        'tpr_dos': tpr_dos,
        'tpr_ddos': tpr_ddos,
        'tpr_web': tpr_web,
        'tpr_bot': tpr_bot,
        'tpr_infil': tpr_infil,
        'tpr_all': tpr_all,
    })

# Output JSON
with open('/tmp/roc_operating_curve.json', 'w') as f:
    json.dump(results_table, f, indent=2)

print(f"{'Threshold':<10} | {'Thursday FAR (%)':<17} | {'Thursday FP Alarms':<18} | {'DoS TPR':<9} | {'DDoS TPR':<9} | {'WebAtk TPR':<11} | {'Bot TPR':<9} | {'Infil TPR':<10} | {'Overall Attack TPR':<18}")
print("-" * 125)
for r in results_table:
    print(f"{r['threshold']:<10.2f} | {r['far_thurs']:>15.2f}% | {r['fp_count']:>10,} / {r['n_thurs']:,} | {r['tpr_dos']:>7.2f}% | {r['tpr_ddos']:>7.2f}% | {r['tpr_web']:>9.2f}% | {r['tpr_bot']:>7.2f}% | {r['tpr_infil']:>8.2f}% | {r['tpr_all']:>16.2f}%")
