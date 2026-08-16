#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Feature-wise (Column-wise) Mean/Std Normalization
Replacing full covariance whitening/rotation (CORAL) with 1D marginal rescaling.

Goal:
1) Calculate target_mean and target_std on CICIDS2017 Thursday Benign Target EVE.
2) Calculate source_mean and source_std on Source Training EVE.
3) Align target features: X_aligned = (X_target - target_mean) / target_std * source_std + source_mean
   (or direct domain-specific standard scaling: z_target = (X - target_mean)/target_std, z_source = (X - source_mean)/source_std)
4) Train / Evaluate XGBoost with this feature-wise normalization.
5) Measure in a SINGLE evaluation pass:
   - Benign FP on CICIDS2017 Thursday (17,420 flows) -> FAR %
   - Attack Recall on CICIDS2018 per class: DoS, DDoS, WebAttack, Bot, Infiltration -> TPR %
   - Attack Recall on Raw Suricata EVE attack streams (Thursday DoS, Tuesday DDoS, etc.)
   - Optimization of Threshold under FAR < 1.0% constraint.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from trainv8 import extract_features_v7, CLASS_MAP, INV_CLASS_MAP

DATA_ROOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_combined")
SOURCE_EVE = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/eve/full_dataset/eve.json")
THURSDAY_EVE = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/eve/cicids2017_thursday_eve.json")
RAW_PCAP_DIR = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/raw_pcap")

print("=" * 100)
print("EXPERIMENT: FEATURE-WISE (MARGINAL) MEAN/STD NORMALIZATION VS FULL CORAL")
print("=" * 100)

# 1. Load Thursday Target Features to compute Target Mean & Std
print("\n[1/5] Loading Target Traffic (CICIDS2017 Thursday Benign EVE)...")
thurs_feats = []
with open(THURSDAY_EVE, 'r') as f:
    for line in f:
        if not line.strip(): continue
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("event_type") != "flow": continue
        fv = extract_features_v7(ev)
        if fv is not None: thurs_feats.append(fv)

X_target = np.array(thurs_feats, dtype=np.float32)
print(f"  Target shape: {X_target.shape} (17,420 flows, pure benign)")

target_mean = np.mean(X_target, axis=0)
target_std = np.std(X_target, axis=0)
target_std[target_std < 1e-6] = 1.0 # Avoid division by zero for constant features

# 2. Load Source Labeled Dataset (CICIDS2018 per class)
print("\n[2/5] Loading Labeled Source Dataset...")
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
    print(f"  Loaded {cname:<12}: {len(feats):>7,} flows")

X_source = np.vstack(X_list)
y_source = np.concatenate(y_list)

source_mean = np.mean(X_source, axis=0)
source_std = np.std(X_source, axis=0)
source_std[source_std < 1e-6] = 1.0

print(f"  Total Source samples: {len(X_source):,}")

# Split Source into Train & Validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_source, y_source, test_size=0.2, random_state=42, stratify=y_source
)

# 3. Define the Domain Feature-wise Alignment Adapters
class FeatureWiseAdapter:
    """1D Feature-wise Mean and Std shift without covariance rotation."""
    def __init__(self, src_mean, src_std, tgt_mean, tgt_std, mode="standardize"):
        self.src_mean = src_mean
        self.src_std = src_std
        self.tgt_mean = tgt_mean
        self.tgt_std = tgt_std
        self.mode = mode # 'rescale' or 'mean_only' or 'target_std'

    def transform_target(self, X):
        if self.mode == "rescale":
            # Map target feature scale to source feature scale:
            # X_out = (X - mu_T) / sigma_T * sigma_S + mu_S
            return (X - self.tgt_mean) / self.tgt_std * self.src_std + self.src_mean
        elif self.mode == "mean_only":
            # Only shift mean, preserve variance
            return X - self.tgt_mean + self.src_mean
        elif self.mode == "standardize":
            # Standardize by target stats
            return (X - self.tgt_mean) / self.tgt_std
        return X

# 4. Train Model with Source Standardized Features
print("\n[3/5] Training XGBoost on Source Features...")
scaler_src = StandardScaler()
X_train_s = scaler_src.fit_transform(X_train)
X_val_s = scaler_src.transform(X_val)

sample_weight = compute_sample_weight('balanced', y_train)

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
    'verbosity': 0,
}

dtrain = xgb.DMatrix(X_train_s, label=y_train, weight=sample_weight)
dval = xgb.DMatrix(X_val_s, label=y_val)

bst = xgb.train(params, dtrain, num_boost_round=600, evals=[(dval, 'val')],
                early_stopping_rounds=30, verbose_eval=False)

print("  XGBoost Training Complete!")

# 5. Evaluate Multiple Normalization Methods Side by Side
methods = [
    ("Method 1: Raw XGBoost (No Adaptation)", None),
    ("Method 2: Feature-wise Rescale (Mean + Std Shift)", FeatureWiseAdapter(source_mean, source_std, target_mean, target_std, mode="rescale")),
    ("Method 3: Feature-wise Mean Shift (Mean Only)", FeatureWiseAdapter(source_mean, source_std, target_mean, target_std, mode="mean_only")),
]

# Load Raw Suricata Attack Files for validation
raw_suricata_attacks = [
    ("DoS (Thu-15)", RAW_PCAP_DIR / "Thursday-15-02-2018" / "eve_DoS.json"),
    ("DDoS (Tue-20)", RAW_PCAP_DIR / "Tuesday-20-02-2018" / "eve_DDoS.json"),
    ("WebAtk (Thu-22)", RAW_PCAP_DIR / "Thursday-22-02-2018" / "eve_WebAttack.json"),
    ("Bot (Fri-02)", RAW_PCAP_DIR / "Friday-02-03-2018" / "eve_Bot.json"),
]
raw_atk_data = {}
for name, path in raw_suricata_attacks:
    if path.exists():
        feats = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip(): continue
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("event_type") != "flow": continue
                fv = extract_features_v7(ev)
                if fv is not None: feats.append(fv)
                if len(feats) >= 10_000: break
        raw_atk_data[name] = np.array(feats, dtype=np.float32)

print("\n" + "=" * 105)
print("TEST RESULTS ACROSS NORMALIZATION METHODS (Optimized for FAR < 1% on Thursday)")
print("=" * 105)

for method_name, adapter in methods:
    print(f"\nEvaluating: {method_name}")
    print("-" * 105)
    
    # Transform Thursday Target
    if adapter is not None:
        X_target_trans = adapter.transform_target(X_target)
    else:
        X_target_trans = X_target
    
    X_target_s = scaler_src.transform(X_target_trans)
    target_probs = bst.predict(xgb.DMatrix(X_target_s))
    target_p_atk = 1.0 - target_probs[:, 0]
    
    # Find Best Threshold that achieves FAR < 1.0% on Thursday traffic
    best_threshold = None
    best_far = None
    
    # Search threshold grid
    for t in np.linspace(0.50, 0.99, 500):
        far = (target_p_atk >= t).mean() * 100.0
        if far < 1.0:
            best_threshold = float(t)
            best_far = far
            break
            
    if best_threshold is None:
        best_threshold = 0.99
        best_far = (target_p_atk >= 0.99).mean() * 100.0
        
    print(f"  Target Thursday Optimization -> Threshold: {best_threshold:.4f} | Achieved FAR: {best_far:.4f}% ({int((target_p_atk>=best_threshold).sum())}/{len(X_target)})")
    
    # Evaluate Validation Attack Set at this Threshold
    val_probs = bst.predict(xgb.DMatrix(X_val_s))
    val_p_atk = 1.0 - val_probs[:, 0]
    val_preds = (val_p_atk >= best_threshold).astype(int)
    
    rec_dos = (val_preds[y_val == 1] == 1).mean() * 100
    rec_ddos = (val_preds[y_val == 2] == 1).mean() * 100
    rec_web = (val_preds[y_val == 3] == 1).mean() * 100
    rec_infil = (val_preds[y_val == 4] == 1).mean() * 100
    rec_bot = (val_preds[y_val == 5] == 1).mean() * 100
    rec_all_val = (val_preds[y_val > 0] == 1).mean() * 100
    
    # Evaluate Raw Suricata Attack Streams with this adapter at this threshold
    raw_recs = {}
    for name, X_raw in raw_atk_data.items():
        if adapter is not None:
            X_raw_trans = adapter.transform_target(X_raw)
        else:
            X_raw_trans = X_raw
        X_raw_s = scaler_src.transform(X_raw_trans)
        p_raw = bst.predict(xgb.DMatrix(X_raw_s))
        p_raw_atk = 1.0 - p_raw[:, 0]
        raw_recs[name] = (p_raw_atk >= best_threshold).mean() * 100
    
    # Print Summary Table Row
    print(f"\n  [PERFORMANCE SUMMARY @ Threshold={best_threshold:.4f}]")
    print(f"    • Thursday Benign FAR      : {best_far:.2f}%  ({'✅ PASS (<1%)' if best_far < 1.0 else '❌ FAIL (>=1%)'})")
    print(f"    • CICIDS2018 Val DoS TPR   : {rec_dos:.2f}%")
    print(f"    • CICIDS2018 Val DDoS TPR  : {rec_ddos:.2f}%")
    print(f"    • CICIDS2018 Val WebAtk TPR: {rec_web:.2f}%")
    print(f"    • CICIDS2018 Val Bot TPR   : {rec_bot:.2f}%")
    print(f"    • CICIDS2018 Val Infil TPR : {rec_infil:.2f}%")
    print(f"    • CICIDS2018 Overall TPR   : {rec_all_val:.2f}%")
    print(f"    --- Suricata Raw Streams (Real Network PCAP) ---")
    for name, rec in raw_recs.items():
        print(f"    • Suricata {name:<15} TPR: {rec:.2f}%")

print("\n" + "=" * 105)
