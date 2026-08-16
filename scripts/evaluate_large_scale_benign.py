#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-Scale Clean Office Traffic Evaluation & Threshold Operating Matrix
Evaluates model on 300,000+ true holdout benign flows from CICIDS2017 Friday
and all attack validation classes side-by-side.
"""

import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7, CLASS_MAP

BASE_DIR = Path("/run/media/mehmet/siber data1/ai modeli xgboost")
BENIGN_PATH = BASE_DIR / "data/relabeled_cicids2017/eve_Benign.json"
MODEL_PATH = BASE_DIR / "models/ids_model_v10c_final.pkl"

def main():
    print("=" * 115)
    print("LARGE-SCALE BENIGN TEST & DUAL-METRIC OPERATING MATRIX (V10c MODEL)")
    print("=" * 115)

    # 1. Load Model Bundle
    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    scaler = bundle['scaler']

    # 2. Load 300,000+ Holdout Benign Flows
    print("Loading 300,000+ Holdout Clean Benign Flows (skipping first 70k)...")
    t0 = time.time()
    benign_feats = []
    with open(BENIGN_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx < 70_000: continue # Strictly holdout
            if len(benign_feats) >= 350_000: break
            if not line.strip(): continue
            try: ev = json.loads(line)
            except: continue
            if ev.get("event_type") != "flow": continue
            fv = extract_features_v7(ev)
            if fv is not None:
                benign_feats.append(fv)
                if len(benign_feats) % 50_000 == 0:
                    print(f"  Loaded {len(benign_feats):,} holdout benign flows...")

    X_benign = np.array(benign_feats, dtype=np.float32)
    print(f"Total Holdout Benign Flows Loaded: {len(X_benign):,} in {time.time() - t0:.1f}s")

    # 3. Load Validation Attack Datasets
    print("\nLoading Attack Validation Sets...")
    attack_dict = {}
    attack_files = {
        "DoS": BASE_DIR / "data/relabeled_combined/eve_DoS.json",
        "DDoS": BASE_DIR / "data/relabeled_combined/eve_DDoS.json",
        "WebAttack": BASE_DIR / "data/relabeled_combined/eve_WebAttack.json",
        "Infiltration": BASE_DIR / "data/relabeled_combined/eve_Infiltration.json",
        "Bot": BASE_DIR / "data/relabeled_combined/eve_Bot_clean.json",
    }
    
    for atype, fpath in attack_files.items():
        feats = []
        with open(fpath, 'r') as f:
            for idx, line in enumerate(f):
                if idx < 20_000: continue # Holdout portion
                if len(feats) >= 10_000: break
                try: ev = json.loads(line)
                except: continue
                if ev.get("event_type") != "flow": continue
                fv = extract_features_v7(ev)
                if fv is not None: feats.append(fv)
        attack_dict[atype] = np.array(feats, dtype=np.float32)
        print(f"  • {atype:<15}: {len(feats):,} holdout attack flows")

    # 4. Predict on Benign
    print("\nPredicting on 300k+ Benign Flows...")
    X_benign_s = scaler.transform(X_benign)
    p_benign = model.predict(xgb.DMatrix(X_benign_s))
    p_benign_atk = 1.0 - p_benign[:, 0]
    n_benign = len(X_benign)

    # 5. Predict on Attacks
    p_attacks = {}
    for atype, arr in attack_dict.items():
        arr_s = scaler.transform(arr)
        p = model.predict(xgb.DMatrix(arr_s))
        p_attacks[atype] = 1.0 - p[:, 0]

    # 6. Evaluation Table
    print("\n" + "=" * 125)
    print(f"{'Eşik (T)':<10} | {'350k Ofis FAR (%)':<18} | {'Sahte Alarm / Toplam':<22} | {'DoS Recall':<11} | {'DDoS Recall':<12} | {'WebAtk':<9} | {'Bot Recall':<11} | {'Infil':<8} | {'GENEL RECALL':<14}")
    print("-" * 125)

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.95]
    for thr in thresholds:
        fp_cnt = int((p_benign_atk >= thr).sum())
        far = (fp_cnt / n_benign) * 100.0

        recalls = {}
        all_rec = []
        for atype, p_arr in p_attacks.items():
            r = (p_arr >= thr).mean() * 100.0
            recalls[atype] = r
            all_rec.extend((p_arr >= thr).tolist())
        
        gen_rec = np.mean(all_rec) * 100.0

        star = "⭐" if thr in (0.80, 0.84) else "  "
        print(f"{thr:<10.2f} | {far:>16.2f}% | {fp_cnt:>8,} / {n_benign:,} | {recalls['DoS']:>9.2f}% | {recalls['DDoS']:>10.2f}% | {recalls['WebAttack']:>7.2f}% | {recalls['Bot']:>9.2f}% | {recalls['Infiltration']:>6.2f}% | {gen_rec:>12.2f}% {star}")

    print("=" * 125)

if __name__ == "__main__":
    main()
