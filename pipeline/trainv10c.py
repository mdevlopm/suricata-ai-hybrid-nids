#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Train V10c: Fully Cleaned Multi-Source Supervised IDS Training
Features:
- Cleaned CICIDS2018 Bot (37,452 flows - DNS/AWS/DHCP removed)
- Cleaned CTU-13 Bot (120,352 flows - DNS/Ads/Background removed)
- CICIDS2017 Friday Bot + DDoS + PortScan + Benign (60k clean office flows)
- CICIDS2018 Benign, DoS, DDoS, WebAttack, Infiltration
Evaluates the dual-metric operating curve (0.50 - 0.99) on Thursday Benign & CICIDS Attack Validation.
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
THURSDAY_EVE = BASE_DIR / "data/eve/cicids2017_thursday_eve.json"
MODEL_OUT = BASE_DIR / "models/ids_model_v10c_final.pkl"

def load_json_flows(fpath, max_samples=None):
    if not fpath.exists():
        return np.empty((0, 70), dtype=np.float32)
    feats = []
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
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
    print("=" * 80)
    print("IDS PIPELINE V10c: FULLY CLEANED MULTI-SOURCE SUPERVISED TRAINING")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Load Data from Multi-Sources
    X_dict = {c: [] for c in CLASS_MAP}

    # A. CICIDS2018 (Cleaned Bot + Attacks + Benign)
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

    # C. CICIDS2017 Friday Flows
    if DATA_2017.exists():
        print("\n[3/4] Loading CICIDS2017 Friday Flows...")
        p_2017_bot = DATA_2017 / "eve_Bot.json"
        p_2017_ddos = DATA_2017 / "eve_DDoS.json"
        p_2017_pscan = DATA_2017 / "eve_PortScan.json"
        p_2017_benign = DATA_2017 / "eve_Benign.json"

        arr_17_bot = load_json_flows(p_2017_bot, max_samples=30_000)
        arr_17_ddos = load_json_flows(p_2017_ddos, max_samples=30_000)
        arr_17_pscan = load_json_flows(p_2017_pscan, max_samples=30_000)
        arr_17_benign = load_json_flows(p_2017_benign, max_samples=60_000)

        if len(arr_17_bot) > 0:
            X_dict["Bot"].append(arr_17_bot)
            print(f"  • CICIDS2017 Friday Bot     : {len(arr_17_bot):,}")
        if len(arr_17_ddos) > 0:
            X_dict["DDoS"].append(arr_17_ddos)
            print(f"  • CICIDS2017 Friday DDoS    : {len(arr_17_ddos):,}")
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
    print("\nTraining XGBoost V10c Multi-Class Classifier...")
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
        'version': 'v10c_cleaned_all_bot',
        'timestamp': datetime.now().isoformat()
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved V10c Model to: {MODEL_OUT}")

    # 5. Dual-Metric Operating Curve Evaluation (0.50 - 0.99)
    print("\n" + "=" * 105)
    print("OPERATING CURVE EVALUATION (0.50 - 0.99) FOR V10c MODEL")
    print("=" * 105)

    # Load Thursday Benign
    thurs_feats = []
    with open(THURSDAY_EVE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try: ev = json.loads(line)
            except: continue
            if ev.get("event_type") != "flow": continue
            fv = extract_features_v7(ev)
            if fv is not None: thurs_feats.append(fv)

    X_thurs = np.array(thurs_feats, dtype=np.float32)
    X_thurs_s = scaler.transform(X_thurs)
    p_thurs = bst.predict(xgb.DMatrix(X_thurs_s))
    p_thurs_atk = 1.0 - p_thurs[:, 0]
    n_thurs = len(X_thurs)

    # Validation Attack Predictions
    p_val = bst.predict(xgb.DMatrix(X_val_s))
    p_val_atk = 1.0 - p_val[:, 0]

    mask_dos = (y_val == 1)
    mask_ddos = (y_val == 2)
    mask_web = (y_val == 3)
    mask_infil = (y_val == 4)
    mask_bot = (y_val == 5)
    mask_all_atk = (y_val > 0)

    print(f"{'Threshold':<10} | {'Thursday FAR (%)':<17} | {'Thursday FP Alarms':<18} | {'DoS TPR':<9} | {'DDoS TPR':<9} | {'WebAtk TPR':<11} | {'Bot TPR':<9} | {'Infil TPR':<10} | {'Overall Attack TPR':<18}")
    print("-" * 125)

    curve_results = []
    for thr in np.arange(0.50, 1.00, 0.01):
        fp_thurs = int((p_thurs_atk >= thr).sum())
        far_thurs = (fp_thurs / n_thurs) * 100.0

        preds_val = (p_val_atk >= thr).astype(int)
        tpr_dos = (preds_val[mask_dos] == 1).mean() * 100.0 if mask_dos.sum() > 0 else 0.0
        tpr_ddos = (preds_val[mask_ddos] == 1).mean() * 100.0 if mask_ddos.sum() > 0 else 0.0
        tpr_web = (preds_val[mask_web] == 1).mean() * 100.0 if mask_web.sum() > 0 else 0.0
        tpr_bot = (preds_val[mask_bot] == 1).mean() * 100.0 if mask_bot.sum() > 0 else 0.0
        tpr_infil = (preds_val[mask_infil] == 1).mean() * 100.0 if mask_infil.sum() > 0 else 0.0
        tpr_all = (preds_val[mask_all_atk] == 1).mean() * 100.0

        print(f"{thr:<10.2f} | {far_thurs:>15.2f}% | {fp_thurs:>10,} / {n_thurs:,} | {tpr_dos:>7.2f}% | {tpr_ddos:>7.2f}% | {tpr_web:>9.2f}% | {tpr_bot:>7.2f}% | {tpr_infil:>8.2f}% | {tpr_all:>16.2f}%")
        curve_results.append({
            'threshold': float(thr),
            'far_thurs': far_thurs,
            'fp_count': fp_thurs,
            'tpr_dos': tpr_dos,
            'tpr_ddos': tpr_ddos,
            'tpr_web': tpr_web,
            'tpr_bot': tpr_bot,
            'tpr_infil': tpr_infil,
            'tpr_all': tpr_all,
        })

    with open('/tmp/v10c_operating_curve.json', 'w') as f:
        json.dump(curve_results, f, indent=2)

if __name__ == "__main__":
    main()
