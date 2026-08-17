#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Train V10d: Multi-Source Supervised Training with Real Friday LOIC DDoS Flows
Features:
- Real Friday LOIC DDoS (40k samples from the newly isolated 258k pool)
- Cleaned CICIDS2018 Bot + Cleaned CTU-13 Bot (120k) + Friday Bot
- Clean Friday Office Benign (60k) + CICIDS2018 Benign (60k)
- Evaluates on the EXACT SAME 192,942 verified clean holdout office flows
- Side-by-side comparison against V10c Baseline
"""

import sys
import os
import json
import time
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from trainv8 import extract_features_v7, CLASS_MAP

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_2018 = BASE_DIR / "data/relabeled_combined"
DATA_CTU13 = BASE_DIR / "data/relabeled_ctu13"
DATA_2017 = BASE_DIR / "data/relabeled_cicids2017"
MODEL_BASELINE = BASE_DIR / "models/baseline/ids_model_v10c_baseline.pkl"
MODEL_OUT = BASE_DIR / "models/ids_model_v10d_final.pkl"

def load_json_flows(fpath, max_samples=None, skip=0):
    if not fpath.exists():
        return np.empty((0, 70), dtype=np.float32)
    feats = []
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            if idx < skip: continue
            if not line.strip(): continue
            try: ev = json.loads(line)
            except Exception: continue
            if ev.get("event_type") != "flow": continue
            fv = extract_features_v7(ev)
            if fv is not None:
                feats.append(fv)
                if max_samples and len(feats) >= max_samples:
                    break
    return np.array(feats, dtype=np.float32) if feats else np.empty((0, 70), dtype=np.float32)

def main():
    print("=" * 100)
    print("IDS PIPELINE V10d: SUPERVISED TRAINING WITH ISOLATED FRIDAY LOIC DDoS FLOWS")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("=" * 100)

    # 1. Load Data from Multi-Sources
    X_dict = {c: [] for c in CLASS_MAP}

    # A. CICIDS2018
    print("\n[1/4] Loading CICIDS2018 Labeled Data...")
    samples_2018 = {
        "Benign": 60_000,
        "DoS": 40_000,
        "DDoS": 40_000,
        "WebAttack": 40_000,
        "Infiltration": 40_000,
    }
    for cname, max_s in samples_2018.items():
        p = DATA_2018 / f"eve_{cname}.json"
        arr = load_json_flows(p, max_samples=max_s)
        X_dict[cname].append(arr)
        print(f"  • CICIDS2018 {cname:<12}: {len(arr):,}")

    # Load Cleaned CICIDS2018 Bot
    p_2018_bot_clean = DATA_2018 / "eve_Bot_clean.json"
    arr_2018_bot = load_json_flows(p_2018_bot_clean, max_samples=37_452)
    X_dict["Bot"].append(arr_2018_bot)
    print(f"  • Cleaned CICIDS2018 Bot    : {len(arr_2018_bot):,}")

    # B. Cleaned CTU-13 Real Suricata Botnet Flows
    print("\n[2/4] Loading Cleaned CTU-13 Malicious Botnet Flows...")
    ctu_bot_path = DATA_CTU13 / "eve_Bot_clean.json"
    arr_ctu = load_json_flows(ctu_bot_path, max_samples=60_000)
    X_dict["Bot"].append(arr_ctu)
    print(f"  • Cleaned CTU-13 Botnet Flows: {len(arr_ctu):,}")

    # C. CICIDS2017 Friday Flows (Using newly isolated pools)
    print("\n[3/4] Loading Newly Isolated CICIDS2017 Friday Flows...")
    p_2017_bot = DATA_2017 / "eve_Bot.json"
    p_2017_ddos = DATA_2017 / "eve_DDoS.json"
    p_2017_pscan = DATA_2017 / "eve_PortScan.json"
    p_2017_benign = DATA_2017 / "eve_Benign.json"

    arr_17_bot = load_json_flows(p_2017_bot, max_samples=30_000)
    # Take first 40,000 DDoS from the 258k pool for training
    arr_17_ddos = load_json_flows(p_2017_ddos, max_samples=40_000)
    arr_17_pscan = load_json_flows(p_2017_pscan, max_samples=30_000)
    # Take first 60,000 Benign from the 264k pool for training
    arr_17_benign = load_json_flows(p_2017_benign, max_samples=60_000)

    if len(arr_17_bot) > 0:
        X_dict["Bot"].append(arr_17_bot)
        print(f"  • CICIDS2017 Friday Bot     : {len(arr_17_bot):,}")
    if len(arr_17_ddos) > 0:
        X_dict["DDoS"].append(arr_17_ddos)
        print(f"  • CICIDS2017 Friday LOIC DDoS: {len(arr_17_ddos):,}")
    if len(arr_17_pscan) > 0:
        X_dict["DoS"].append(arr_17_pscan)
        print(f"  • CICIDS2017 Friday PortScan: {len(arr_17_pscan):,}")
    if len(arr_17_benign) > 0:
        X_dict["Benign"].append(arr_17_benign)
        print(f"  • CICIDS2017 Friday Benign  : {len(arr_17_benign):,}")

    # Combine all classes
    X_parts, y_parts = [], []
    print("\n--- Composite Dataset Class Breakdown ---")
    for cname, cidx in CLASS_MAP.items():
        if X_dict[cname]:
            merged = np.vstack(X_dict[cname])
            X_parts.append(merged)
            y_parts.append(np.full(len(merged), cidx, dtype=int))
            print(f"  • Class {cidx} ({cname:<12}): {len(merged):,} flows")

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)
    print(f"Total Combined Training Samples: {len(X_all):,}")

    # 2. Train/Validation Split & Scaling
    print("\n[4/4] Preparing Train/Validation Split and Scaler...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    sample_weights = compute_sample_weight('balanced', y_train)

    # 3. XGBoost Model Training
    print("\nTraining XGBoost V10d Multi-Class Classifier...")
    params = {
        'objective': 'multi:softprob',
        'num_class': len(CLASS_MAP),
        'eval_metric': 'mlogloss',
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cpu',
        'seed': 42,
        'verbosity': 1,
    }

    dtrain = xgb.DMatrix(X_train_s, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val_s, label=y_val)

    t0 = time.time()
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=[(dval, 'validation')],
        early_stopping_rounds=30,
        verbose_eval=50
    )
    print(f"Training finished in {time.time() - t0:.1f}s")

    # 4. Save Model Bundle
    bundle = {
        'model': bst,
        'scaler': scaler,
        'class_map': CLASS_MAP,
        'coral_adapter': None,
        'threshold': 0.84,
        'version': 'v10d_loic_ddos_enriched',
        'timestamp': datetime.now().isoformat()
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved V10d Model to: {MODEL_OUT}")

    # 5. Load EXACT SAME 192,942 Clean Holdout Office Flows
    print("\n" + "=" * 115)
    print("BENCHMARK: EVALUATION ON EXACT SAME 192,942 VERIFIED CLEAN OFFICE FLOWS")
    print("=" * 115)

    print("Loading the 192,942 verified clean holdout flows (skipping first 70k)...")
    clean_holdout_feats = []
    with open(p_2017_benign, 'r') as f:
        for idx, line in enumerate(f):
            if idx < 70_000: continue # Strictly holdout
            if len(clean_holdout_feats) >= 192_942: break
            try: ev = json.loads(line)
            except: continue
            fv = extract_features_v7(ev)
            if fv is not None:
                clean_holdout_feats.append(fv)

    X_clean_test = np.array(clean_holdout_feats, dtype=np.float32)
    n_clean = len(X_clean_test)
    print(f"Loaded {n_clean:,} Verified Clean Holdout Flows.")

    # 6. Load Validation Attack Datasets (Holdout Portions)
    print("\nLoading Attack Validation Sets...")
    attack_dict = {}
    attack_files = {
        "DoS": BASE_DIR / "data/relabeled_combined/eve_DoS.json",
        "DDoS": BASE_DIR / "data/relabeled_combined/eve_DDoS.json",
        "WebAttack": BASE_DIR / "data/relabeled_combined/eve_WebAttack.json",
        "Infiltration": BASE_DIR / "data/relabeled_combined/eve_Infiltration.json",
        "Bot": BASE_DIR / "data/relabeled_combined/eve_Bot_clean.json",
        "Friday_DDoS": p_2017_ddos, # 10k holdout from Friday LOIC pool
    }
    
    for atype, fpath in attack_files.items():
        feats = []
        skip_n = 50_000 if atype == "Friday_DDoS" else 20_000
        with open(fpath, 'r') as f:
            for idx, line in enumerate(f):
                if idx < skip_n: continue
                if len(feats) >= 10_000: break
                try: ev = json.loads(line)
                except: continue
                if ev.get("event_type") != "flow": continue
                fv = extract_features_v7(ev)
                if fv is not None: feats.append(fv)
        attack_dict[atype] = np.array(feats, dtype=np.float32)
        print(f"  • {atype:<15}: {len(feats):,} holdout attack flows")

    # 7. Load Baseline V10c for Side-by-Side Comparison
    with open(MODEL_BASELINE, 'rb') as f:
        bundle_v10c = pickle.load(f)
    model_v10c = bundle_v10c['model']
    scaler_v10c = bundle_v10c['scaler']

    # Predict V10d
    p_clean_v10d = bst.predict(xgb.DMatrix(scaler.transform(X_clean_test)))
    p_clean_v10d_atk = 1.0 - p_clean_v10d[:, 0]

    # Predict V10c
    p_clean_v10c = model_v10c.predict(xgb.DMatrix(scaler_v10c.transform(X_clean_test)))
    p_clean_v10c_atk = 1.0 - p_clean_v10c[:, 0]

    print("\n" + "=" * 135)
    print("SIDE-BY-SIDE COMPARISON: V10c BASELINE vs V10d CANDIDATE (ON EXACT SAME 192k CLEAN SET)")
    print("=" * 135)
    print(f"{'Eşik (T)':<8} | {'V10c FAR':<10} | {'V10d FAR':<10} | {'DoS (V10c / V10d)':<19} | {'DDoS (V10c / V10d)':<20} | {'Friday DDoS (V10c/V10d)':<24} | {'Bot (V10c / V10d)':<19} | {'GENEL RECALL (V10c/V10d)':<25}")
    print("-" * 135)

    thresholds = [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.90]
    for thr in thresholds:
        # FAR
        far_c = (p_clean_v10c_atk >= thr).mean() * 100.0
        far_d = (p_clean_v10d_atk >= thr).mean() * 100.0

        # Recalls
        rec_c, rec_d = {}, {}
        for atype, arr in attack_dict.items():
            pc = model_v10c.predict(xgb.DMatrix(scaler_v10c.transform(arr)))
            pd = bst.predict(xgb.DMatrix(scaler.transform(arr)))
            rec_c[atype] = (1.0 - pc[:, 0] >= thr).mean() * 100.0
            rec_d[atype] = (1.0 - pd[:, 0] >= thr).mean() * 100.0

        gen_c = np.mean([rec_c['DoS'], rec_c['DDoS'], rec_c['WebAttack'], rec_c['Bot'], rec_c['Infiltration']])
        gen_d = np.mean([rec_d['DoS'], rec_d['DDoS'], rec_d['WebAttack'], rec_d['Bot'], rec_d['Infiltration']])

        star = "⭐" if thr == 0.84 else "  "
        print(f"{thr:<8.2f} | {far_c:>8.2f}% | {far_d:>8.2f}% | {rec_c['DoS']:>6.1f}% / {rec_d['DoS']:<6.1f}% | {rec_c['DDoS']:>6.1f}% / {rec_d['DDoS']:<6.1f}%  | {rec_c['Friday_DDoS']:>9.1f}% / {rec_d['Friday_DDoS']:<9.1f}% | {rec_c['Bot']:>6.1f}% / {rec_d['Bot']:<6.1f}% | {gen_c:>9.2f}% / {gen_d:<9.2f}% {star}")

    print("=" * 135)

if __name__ == "__main__":
    main()
