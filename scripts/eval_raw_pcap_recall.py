#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate raw PCAP EVE files in data/raw_pcap/ with models/ids_model_v8_final.pkl
Testing both:
- Target Domain Mode (with CORAL transform_target -> Scaler -> XGBoost @ 0.84)
- Source Domain Mode (Direct Scaler -> XGBoost @ 0.84)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np

from trainv8 import extract_features_v7

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/ids_model_v8_final.pkl"
RAW_PCAP_DIR = BASE_DIR / "data/raw_pcap"

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
threshold = float(bundle.get('threshold', 0.84))
coral = bundle['coral_adapter']

raw_test_files = [
    ("DoS (Thu-15-02)", RAW_PCAP_DIR / "Thursday-15-02-2018" / "eve_DoS.json"),
    ("DoS (Fri-16-02)", RAW_PCAP_DIR / "Friday-16-02-2018" / "eve_DoS.json"),
    ("DDoS (Tue-20-02)", RAW_PCAP_DIR / "Tuesday-20-02-2018" / "eve_DDoS.json"),
    ("DDoS (Wed-21-02)", RAW_PCAP_DIR / "Wednesday-21-02-2018" / "eve_DDoS.json"),
    ("WebAttack (Thu-22-02)", RAW_PCAP_DIR / "Thursday-22-02-2018" / "eve_WebAttack.json"),
    ("WebAttack (Fri-23-02)", RAW_PCAP_DIR / "Friday-23-02-2018" / "eve_WebAttack.json"),
    ("Bot (Fri-02-03)", RAW_PCAP_DIR / "Friday-02-03-2018" / "eve_Bot.json"),
    ("Infiltration (Wed-28-02)", RAW_PCAP_DIR / "Wednesday-28-02-2018" / "eve_Infiltration.json"),
]

print("=" * 80)
print(f"EVALUATING RAW PCAP SURICATA EVE ATTACK FILES (Threshold = {threshold})")
print("=" * 80)
print(f"{'Attack File / Dataset':<28} | {'Flows':>8} | {'Raw Scaled Recall':>18} | {'With CORAL Recall':>18}")
print("-" * 80)

for label, fpath in raw_test_files:
    if not fpath.exists():
        print(f"  Missing: {fpath}")
        continue
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
            if len(feats) >= 10_000: # Fast evaluation on 10k flows per raw file
                break
    
    if not feats:
        print(f"{label:<28} | {'0':>8} | {'N/A':>18} | {'N/A':>18}")
        continue
    
    X = np.array(feats, dtype=np.float32)
    
    # 1) Direct Scaled
    X_s = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        p_raw = model.predict_proba(X_s)
    else:
        import xgboost as xgb
        p_raw = model.predict(xgb.DMatrix(X_s))
    rec_raw = ((1.0 - p_raw[:, 0]) >= threshold).mean()
    
    # 2) With CORAL transform_target
    if coral and coral.is_fitted_:
        X_coral = coral.transform_target(X)
        X_coral_s = scaler.transform(X_coral)
        if hasattr(model, "predict_proba"):
            p_coral = model.predict_proba(X_coral_s)
        else:
            import xgboost as xgb
            p_coral = model.predict(xgb.DMatrix(X_coral_s))
        rec_coral = ((1.0 - p_coral[:, 0]) >= threshold).mean()
        rec_coral_str = f"{rec_coral*100:.2f}%"
    else:
        rec_coral_str = "N/A"
    
    print(f"{label:<28} | {len(feats):>8,} | {rec_raw*100:>17.2f}% | {rec_coral_str:>18}")

print("=" * 80)
