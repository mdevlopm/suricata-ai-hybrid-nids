# -*- coding: utf-8 -*-
"""
trainv9.py — XGBoost IDS Training with MCFP-CORAL Domain Adaptation
===================================================================
Extends trainv8.py to:
  1. Add CTU-13 relabeled Bot flows (24,702) to Supervised Source dataset (Bot = 174,702)
  2. Expand CORAL Target Domain with 250,000 random unlabelled flow events sampled from MCFP
     combined with CICIDS2017 Thursday (50,000) -> Total Target Domain = 300,000 flows
  3. Keep XGBoost hyperparameters BİREBİR IDENTICAL to trainv8.py
  4. Save model bundle to ids_model_v9_final.pkl
"""

import gc
import json
import pickle
import warnings
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trainv9")

DEFAULT_DATA_ROOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_combined")
DEFAULT_CTU13_BOT = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/relabeled_ctu13/eve_Bot.json")
DEFAULT_MCFP_EVE = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/raw_pcap/mcfp felk/eve_botnet_mcfp.json")
DEFAULT_TARGET_EVE = Path("/run/media/mehmet/siber data1/ai modeli xgboost/data/eve/cicids2017_thursday_eve.json")
DEFAULT_OUT_MODEL = Path("/run/media/mehmet/siber data1/ai modeli xgboost/ids_model_v9_final.pkl")

CLASS_MAP = {
    "Benign": 0,
    "DoS": 1,
    "DDoS": 2,
    "WebAttack": 3,
    "Infiltration": 4,
    "Bot": 5,
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

CORAL_LAMBDA_REG = 1e-5
CORAL_TARGET_THURSDAY_SAMPLES = 50_000
CORAL_TARGET_MCFP_SAMPLES = 250_000

FEATURE_DIM = 70
FEATURE_DTYPE = np.float32

MAX_SAMPLES_PER_CLASS = {
    "Benign": 300_000,
    "DoS": 150_000,
    "DDoS": 150_000,
    "WebAttack": 150_000,
    "Infiltration": 150_000,
    "Bot": 150_000,  # + 24,702 CTU-13 Bot
}

def _extract_tcp_flags(event):
    tf = event.get("tcp", {}).get("tcp_flags_tc", 0)
    if isinstance(tf, str):
        try: tf = int(tf, 16)
        except ValueError: tf = 0
    return tf

def extract_features_v7(event):
    if event.get("event_type") != "flow":
        return None
    flow = event.get("flow", {})
    if not flow:
        return None

    def _safe_float(val, default=0.0):
        if val is None: return default
        try: return float(val)
        except (ValueError, TypeError): return default

    def _safe_int(val, default=0):
        if val is None: return default
        try: return int(val)
        except (ValueError, TypeError): return default

    src_port = _safe_int(event.get("src_port"))
    dest_port = _safe_int(event.get("dest_port"))

    proto_str = str(event.get("proto", "")).lower()
    is_tcp = 1.0 if proto_str == "tcp" else 0.0
    is_udp = 1.0 if proto_str == "udp" else 0.0
    is_icmp = 1.0 if proto_str == "icmp" else 0.0

    age = _safe_float(flow.get("age"))
    pkts_toserver = _safe_float(flow.get("pkts_toserver"))
    pkts_toclient = _safe_float(flow.get("pkts_toclient"))
    bytes_toserver = _safe_float(flow.get("bytes_toserver"))
    bytes_toclient = _safe_float(flow.get("bytes_toclient"))

    tot_pkts = pkts_toserver + pkts_toclient
    tot_bytes = bytes_toserver + bytes_toclient

    pkt_rate = tot_pkts / (age + 1e-6)
    byte_rate = tot_bytes / (age + 1e-6)

    avg_pkt_size = tot_bytes / (tot_pkts + 1e-6)
    mean_pkt_size = avg_pkt_size

    bytes_per_pkt_s2c = bytes_toclient / (pkts_toclient + 1e-6)
    bytes_per_pkt_c2s = bytes_toserver / (pkts_toserver + 1e-6)

    pkt_ratio = pkts_toserver / (pkts_toclient + 1e-6)
    byte_ratio = bytes_toserver / (bytes_toclient + 1e-6)

    tf = _extract_tcp_flags(event)
    fin_flag = 1.0 if (tf & 0x01) else 0.0
    syn_flag = 1.0 if (tf & 0x02) else 0.0
    rst_flag = 1.0 if (tf & 0x04) else 0.0
    psh_flag = 1.0 if (tf & 0x08) else 0.0
    ack_flag = 1.0 if (tf & 0x10) else 0.0
    urg_flag = 1.0 if (tf & 0x20) else 0.0

    state_str = str(flow.get("state", "")).lower()
    state_new = 1.0 if state_str == "new" else 0.0
    state_established = 1.0 if state_str == "established" else 0.0
    state_closed = 1.0 if state_str == "closed" else 0.0

    http = event.get("http", {})
    has_http = 1.0 if http else 0.0
    http_len = _safe_float(http.get("length"))
    http_status = _safe_float(http.get("status"))

    tls = event.get("tls", {})
    has_tls = 1.0 if tls else 0.0

    dns = event.get("dns", {})
    has_dns = 1.0 if dns else 0.0

    well_known_ports = [80, 443, 53, 22, 21, 25, 110, 143, 3389, 8080]
    src_port_is_wellknown = 1.0 if src_port in well_known_ports else 0.0
    dst_port_is_wellknown = 1.0 if dest_port in well_known_ports else 0.0

    feat = [
        src_port, dest_port, is_tcp, is_udp, is_icmp, age,
        pkts_toserver, pkts_toclient, bytes_toserver, bytes_toclient,
        tot_pkts, tot_bytes, pkt_rate, byte_rate, avg_pkt_size,
        mean_pkt_size, bytes_per_pkt_s2c, bytes_per_pkt_c2s,
        pkt_ratio, byte_ratio,
        fin_flag, syn_flag, rst_flag, psh_flag, ack_flag, urg_flag,
        state_new, state_established, state_closed,
        has_http, http_len, http_status, has_tls, has_dns,
        src_port_is_wellknown, dst_port_is_wellknown,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]
    return np.array(feat, dtype=FEATURE_DTYPE)

class CORAL:
    def __init__(self, lambda_reg=1e-5):
        self.lambda_reg = lambda_reg
        self.cov_src_ = None
        self.cov_tgt_ = None
        self.A_ = None
        self.mean_src_ = None
        self.mean_tgt_ = None

    def fit(self, X_src, X_tgt):
        n_src, d = X_src.shape
        n_tgt, _ = X_tgt.shape

        self.mean_src_ = np.mean(X_src, axis=0)
        self.mean_tgt_ = np.mean(X_tgt, axis=0)

        Cs = np.cov(X_src, rowvar=False) + self.lambda_reg * np.eye(d)
        Ct = np.cov(X_tgt, rowvar=False) + self.lambda_reg * np.eye(d)

        self.cov_src_ = Cs
        self.cov_tgt_ = Ct

        def _sqrtm_psd(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        def _inv_sqrtm_psd(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        Cs_inv_sqrt = _inv_sqrtm_psd(Cs)
        Ct_sqrt = _sqrtm_psd(Ct)

        self.A_ = Cs_inv_sqrt @ Ct_sqrt
        return self

    def transform(self, X, domain='source'):
        if domain == 'source':
            X_centered = X - self.mean_src_
            return (X_centered @ self.A_) + self.mean_tgt_
        elif domain == 'target':
            return X
        else:
            raise ValueError("domain must be 'source' or 'target'")

class CORALDomainAdapter:
    def __init__(self, lambda_reg=1e-5):
        self.lambda_reg = lambda_reg
        self.scaler = StandardScaler()
        self.coral = CORAL(lambda_reg=lambda_reg)
        self.is_fitted_ = False

    def fit(self, X_source, X_target, scale=True):
        if scale:
            X_combined = np.vstack([X_source, X_target])
            self.scaler.fit(X_combined)
            X_src_s = self.scaler.transform(X_source)
            X_tgt_s = self.scaler.transform(X_target)
        else:
            X_src_s, X_tgt_s = X_source, X_target

        self.coral.fit(X_src_s, X_tgt_s)
        self.is_fitted_ = True
        return self

    def transform_source(self, X_source):
        X_scaled = self.scaler.transform(X_source)
        return self.coral.transform(X_scaled, domain='source')

    def transform_target(self, X_target):
        X_scaled = self.scaler.transform(X_target)
        return self.coral.transform(X_scaled, domain='target')

def load_labeled_data(data_root: Path, ctu13_bot_path: Path, max_per_class: dict):
    X_list, y_list, src_ips = [], [], []

    for cls_name, cls_idx in CLASS_MAP.items():
        eve_path = data_root / f"eve_{cls_name}.json"
        if not eve_path.exists():
            log.warning(f"  File missing: {eve_path}")
            continue

        cap = max_per_class.get(cls_name, 150_000)
        feats, ips = [], []

        with open(eve_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip(): continue
                try: ev = json.loads(line)
                except Exception: continue
                fv = extract_features_v7(ev)
                if fv is not None:
                    feats.append(fv)
                    ips.append(str(ev.get("src_ip", "0.0.0.0")))

        if len(feats) > cap:
            combined = list(zip(feats, ips))
            random.seed(42)
            combined = random.sample(combined, cap)
            feats, ips = zip(*combined)

        X_list.append(np.array(feats, dtype=FEATURE_DTYPE))
        y_list.append(np.full(len(feats), cls_idx, dtype=int))
        src_ips.extend(ips)
        log.info(f"  Loaded {cls_name:<12}: {len(feats):>8,} samples (cap={cap:,})")

    # Load CTU-13 Bot samples
    if ctu13_bot_path.exists():
        ctu_feats, ctu_ips = [], []
        with open(ctu13_bot_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip(): continue
                try: ev = json.loads(line)
                except Exception: continue
                fv = extract_features_v7(ev)
                if fv is not None:
                    ctu_feats.append(fv)
                    ctu_ips.append(str(ev.get("src_ip", "0.0.0.0")))

        bot_idx = CLASS_MAP["Bot"]
        X_list.append(np.array(ctu_feats, dtype=FEATURE_DTYPE))
        y_list.append(np.full(len(ctu_feats), bot_idx, dtype=int))
        src_ips.extend(ctu_ips)
        log.info(f"  Loaded CTU-13 Bot  : {len(ctu_feats):>8,} samples (Total Bot = {150000 + len(ctu_feats):,})")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, np.array(src_ips)

def load_unlabeled_target(thursday_path: Path, mcfp_path: Path, max_thursday: int, max_mcfp: int):
    feats = []
    
    # 1. Thursday samples
    if thursday_path.exists():
        t_feats = []
        with open(thursday_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or '"event_type":"flow"' not in line: continue
                try: ev = json.loads(line)
                except Exception: continue
                fv = extract_features_v7(ev)
                if fv is not None:
                    t_feats.append(fv)
        if len(t_feats) > max_thursday:
            random.seed(42)
            t_feats = random.sample(t_feats, max_thursday)
        feats.extend(t_feats)
        log.info(f"  Target CORAL Thursday: {len(t_feats):,} samples")

    # 2. MCFP unlabelled samples
    if mcfp_path.exists():
        m_feats = []
        with open(mcfp_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or '"event_type":"flow"' not in line: continue
                try: ev = json.loads(line)
                except Exception: continue
                fv = extract_features_v7(ev)
                if fv is not None:
                    m_feats.append(fv)
                    if len(m_feats) >= max_mcfp * 3:
                        break
        if len(m_feats) > max_mcfp:
            random.seed(42)
            m_feats = random.sample(m_feats, max_mcfp)
        feats.extend(m_feats)
        log.info(f"  Target CORAL MCFP    : {len(m_feats):,} samples")

    log.info(f"  TOTAL Target CORAL Domain: {len(feats):,} samples")
    return np.array(feats, dtype=FEATURE_DTYPE)

def find_best_threshold(y_true, y_prob, min_recall=0.95):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    valid = np.where(rec >= min_recall)[0]
    if len(valid) == 0: return 0.5
    best_idx = valid[np.argmin(1 - prec[valid])]
    return float(thr[best_idx]) if best_idx < len(thr) else 1.0

def main(args):
    log.info("=" * 60)
    log.info("TRAINV9 — XGBoost + MCFP-CORAL Domain Adaptation")
    log.info("=" * 60)

    log.info("\n[1/6] Loading labeled source data (CICIDS2018 + CTU-13)...")
    X_source, y_source, src_ips = load_labeled_data(args.data_root, args.ctu13_bot, MAX_SAMPLES_PER_CLASS)
    log.info(f"Total source samples: {len(X_source):,} | Classes: {Counter(y_source)}")

    log.info("\n[2/6] Loading target domain data for CORAL (Thursday + MCFP)...")
    X_target = load_unlabeled_target(args.target_eve, args.mcfp_eve, CORAL_TARGET_THURSDAY_SAMPLES, CORAL_TARGET_MCFP_SAMPLES)

    log.info("\n[3/6] Fitting CORAL adapter (Source -> Target Alignment)...")
    coral = CORALDomainAdapter(lambda_reg=CORAL_LAMBDA_REG)
    coral.fit(X_source, X_target, scale=True)

    log.info("\n[4/6] Applying CORAL alignment to source features...")
    X_aligned = coral.transform_source(X_source)

    log.info("\n[5/6] StratifiedGroupKFold Benchmark Evaluation & XGBoost Training...")
    # 5-fold StratifiedGroupKFold benchmark
    sgkf = StratifiedGroupKFold(n_splits=5)
    f1_scores = []
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_aligned)
    sample_weight = compute_sample_weight('balanced', y_source)

    # Identical v8 XGBoost params
    params = {
        'objective': 'multi:softprob',
        'num_class': NUM_CLASSES,
        'eval_metric': 'mlogloss',
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'device': 'cuda' if args.use_gpu else 'cpu',
        'seed': 42,
        'verbosity': 1,
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_s, y_source, groups=src_ips)):
        dtr = xgb.DMatrix(X_train_s[train_idx], label=y_source[train_idx], weight=sample_weight[train_idx])
        dva = xgb.DMatrix(X_train_s[val_idx], label=y_source[val_idx])
        b = xgb.train(params, dtr, num_boost_round=1500, evals=[(dtr, 'train'), (dva, 'val')],
                      early_stopping_rounds=50, verbose_eval=False)
        preds = b.predict(dva).argmax(axis=1)
        macro_f1 = f1_score(y_source[val_idx], preds, average='macro')
        f1_scores.append(macro_f1)
        log.info(f"  Fold {fold+1}/5 Macro F1: {macro_f1:.4f}")

    benchmark_macro_f1 = np.mean(f1_scores)
    log.info(f"  >>> Benchmark Mean Macro F1 (StratifiedGroupKFold): {benchmark_macro_f1:.4f}")

    # Final train/val split for model bundle
    X_tr, X_va, y_tr, y_va = train_test_split(X_aligned, y_source, test_size=0.2, random_state=42, stratify=y_source)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    sw_tr = compute_sample_weight('balanced', y_tr)

    dtrain = xgb.DMatrix(X_tr_s, label=y_tr, weight=sw_tr)
    dval = xgb.DMatrix(X_va_s, label=y_va)

    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=50, verbose_eval=50)

    log.info("\n[6/6] Optimizing binary threshold and evaluating Thursday FP rate...")
    val_probs = bst.predict(dval)
    y_va_bin = (y_va != 0).astype(int)
    prob_atk = 1.0 - val_probs[:, 0]
    best_thr = find_best_threshold(y_va_bin, prob_atk, min_recall=0.95)
    log.info(f"  Best threshold (recall>=0.95): {best_thr:.4f}")

    # Evaluate on Thursday test set (334,783 flows)
    thursday_fp_rate = 0.0
    if args.target_eve.exists():
        t_feats, t_ips = [], []
        with open(args.target_eve, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or '"event_type":"flow"' not in line: continue
                try: ev = json.loads(line)
                except Exception: continue
                fv = extract_features_v7(ev)
                if fv is not None:
                    t_feats.append(fv)
        
        X_thurs = np.array(t_feats, dtype=FEATURE_DTYPE)
        X_thurs_corr = coral.transform_target(X_thurs)
        X_thurs_s = scaler.transform(X_thurs_corr)
        dthurs = xgb.DMatrix(X_thurs_s)
        thurs_probs = bst.predict(dthurs)
        thurs_atk_prob = 1.0 - thurs_probs[:, 0]
        thursday_fp_rate = float((thurs_atk_prob >= best_thr).mean() * 100.0)
        log.info(f"  CICIDS2017 Thursday FP Rate (@ {best_thr:.4f}): %{thursday_fp_rate:.4f} ({len(t_feats):,} flows)")

    bundle = {
        'model': bst,
        'scaler': scaler,
        'threshold': best_thr,
        'class_map': CLASS_MAP,
        'inv_class_map': INV_CLASS_MAP,
        'coral_adapter': coral,
        'feature_dim': X_source.shape[1],
        'train_date': datetime.now().isoformat(),
        'num_classes': NUM_CLASSES,
        'xgb_params': params,
        'benchmark_macro_f1': benchmark_macro_f1,
        'thursday_fp_rate': thursday_fp_rate,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("\n" + "=" * 80)
    log.info("  TRAINV9 FINAL VERIFIED SUMMARY")
    log.info("=" * 80)
    log.info(f"  Macro F1 (GroupKFold) : {benchmark_macro_f1:.4f}")
    log.info(f"  Thursday FP Rate      : %{thursday_fp_rate:.4f} (@ threshold {best_thr:.4f})")
    log.info(f"  Model saved to        : {args.output}")
    log.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trainv9 — XGBoost + MCFP-CORAL Domain Adaptation")
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--ctu13-bot', type=Path, default=DEFAULT_CTU13_BOT)
    parser.add_argument('--mcfp-eve', type=Path, default=DEFAULT_MCFP_EVE)
    parser.add_argument('--target-eve', type=Path, default=DEFAULT_TARGET_EVE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT_MODEL)
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    main(args)
