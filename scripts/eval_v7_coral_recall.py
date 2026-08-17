#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Recall / TPR and FP rate of:
- v7_final (archive/models/ids_model_v7_final.pkl) + Live CORAL adapter (pipeline/coral_adapter.pkl)
- Compare with v7_final WITHOUT CORAL
- Evaluate on:
  1) CICIDS2018 Validation Set (DoS, DDoS, WebAttack, Bot, Infiltration, Benign)
  2) Raw PCAP Suricata EVE attack files
  3) CICIDS2017 Thursday Benign EVE (FP rate)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split

from trainv8 import extract_features_v7, CLASS_MAP
from coral_domain_adaptation import CORALDomainAdapter

BASE_DIR = Path(__file__).resolve().parent.parent
V7_MODEL_PATH = BASE_DIR / "archive/models/ids_model_v7_final.pkl"
CORAL_PATH = BASE_DIR / "pipeline/coral_adapter.pkl"
DATA_ROOT = BASE_DIR / "data/relabeled_combined"
THURSDAY_EVE = BASE_DIR / "data/eve/cicids2017_thursday_eve.json"

print("=" * 90)
print("1. LOADING v7 MODEL & LIVE CORAL ADAPTER")
print("=" * 90)

with open(V7_MODEL_PATH, 'rb') as f:
    bundle_v7 = pickle.load(f)

model_v7 = bundle_v7['model']
scaler_v7 = bundle_v7['scaler']
thresh_v7 = float(bundle_v7.get('threshold', 0.66))

with open(CORAL_PATH, 'rb') as f:
    coral_live = pickle.load(f)

print(f"Model v7 Threshold : {thresh_v7:.4f} (Production spec: 0.66)")
print(f"CORAL Live Fitted  : {coral_live.is_fitted_}")

# Load 25k per class for validation
print("\n[2] Loading CICIDS2018 validation dataset...")
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
            if len(feats) >= 25_000: break
    X_list.append(np.array(feats, dtype=np.float32))
    y_list.append(np.full(len(feats), cidx, dtype=int))
    print(f"  Loaded {cname:<12}: {len(feats):>7,} flows")

X_all = np.vstack(X_list)
y_all = np.concatenate(y_list)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ----------------------------------------------------------------------------
# 1) v7 WITHOUT CORAL (Raw Scaled @ 0.66)
# ----------------------------------------------------------------------------
X_val_s_raw = scaler_v7.transform(X_val)
if hasattr(model_v7, 'predict_proba'):
    probs_v7_raw = model_v7.predict_proba(X_val_s_raw)
else:
    import xgboost as xgb
    probs_v7_raw = model_v7.predict(xgb.DMatrix(X_val_s_raw))

preds_v7_raw = ((1.0 - probs_v7_raw[:, 0]) >= thresh_v7).astype(int)

# ----------------------------------------------------------------------------
# 2) v7 WITH LIVE CORAL (CORAL transform_target -> Scaler @ 0.66)
# ----------------------------------------------------------------------------
X_val_coral = coral_live.transform_target(X_val)
X_val_s_coral = scaler_v7.transform(X_val_coral)
if hasattr(model_v7, 'predict_proba'):
    probs_v7_coral = model_v7.predict_proba(X_val_s_coral)
else:
    import xgboost as xgb
    probs_v7_coral = model_v7.predict(xgb.DMatrix(X_val_s_coral))

preds_v7_coral = ((1.0 - probs_v7_coral[:, 0]) >= thresh_v7).astype(int)

# ----------------------------------------------------------------------------
# 3) FP Rate on Thursday Target Benign Traffic
# ----------------------------------------------------------------------------
print("\n[3] Evaluating FP on Thursday Benign EVE (17,420 flows)...")
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

# Raw v7 on Thursday
X_thurs_s_raw = scaler_v7.transform(X_thurs)
p_thurs_raw = model_v7.predict_proba(X_thurs_s_raw) if hasattr(model_v7, 'predict_proba') else model_v7.predict(xgb.DMatrix(X_thurs_s_raw))
thurs_fp_raw = ((1.0 - p_thurs_raw[:, 0]) >= thresh_v7).mean() * 100

# v7 + CORAL on Thursday
X_thurs_coral = coral_live.transform_target(X_thurs)
X_thurs_s_coral = scaler_v7.transform(X_thurs_coral)
p_thurs_coral = model_v7.predict_proba(X_thurs_s_coral) if hasattr(model_v7, 'predict_proba') else model_v7.predict(xgb.DMatrix(X_thurs_s_coral))
thurs_fp_coral = ((1.0 - p_thurs_coral[:, 0]) >= thresh_v7).mean() * 100

print("\n" + "=" * 90)
print("RESULTS: v7 MODEL + LIVE CORAL RECALL & FP COMPARISON")
print("=" * 90)
print(f"{'Class / Dataset':<25} | {'Samples':>8} | {'v7 WITHOUT CORAL (0.66)':>24} | {'v7 WITH LIVE CORAL (0.66)':>25}")
print("-" * 90)

for cname, cidx in CLASS_MAP.items():
    mask = (y_val == cidx)
    n = mask.sum()
    if cidx == 0:
        far_raw = (preds_v7_raw[mask] == 1).mean() * 100
        far_cor = (preds_v7_coral[mask] == 1).mean() * 100
        print(f"{cname + ' (Source Val)':<25} | {n:>8,} | {'FAR: ' + f'{far_raw:.2f}%':>24} | {'FAR: ' + f'{far_cor:.2f}%':>25}")
    else:
        rec_raw = (preds_v7_raw[mask] == 1).mean() * 100
        rec_cor = (preds_v7_coral[mask] == 1).mean() * 100
        print(f"{cname:<25} | {n:>8,} | {rec_raw:>23.2f}% | {rec_cor:>24.2f}%")

# Overall Attack Recall
atk_mask = (y_val > 0)
rec_all_raw = (preds_v7_raw[atk_mask] == 1).mean() * 100
rec_all_cor = (preds_v7_coral[atk_mask] == 1).mean() * 100
print("-" * 90)
print(f"{'OVERALL ATTACK RECALL':<25} | {atk_mask.sum():>8,} | {rec_all_raw:>23.2f}% | {rec_all_cor:>24.2f}%")
print(f"{'Thursday BENIGN FP RATE':<25} | {len(X_thurs):>8,} | {thurs_fp_raw:>23.2f}% | {thurs_fp_coral:>24.2f}%")
print("=" * 90)
