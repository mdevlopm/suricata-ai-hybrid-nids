# -*- coding: utf-8 -*-
"""
trainv8.py — XGBoost IDS Training with CORAL Domain Adaptation
================================================================
Extends trainv7.py to:
  1. Accept relabeled eve_<Type>.json files (Benign/DoS/DDoS/WebAttack/Bot/Infiltration)
  2. Apply CORAL alignment (source→target) before training
  3. Save CORAL adapter alongside model for inference pipeline

Pipeline:
1. Read eve_<Type>.json files per class (relabeled by relabel_cicids2018.py)
2. Extract 70 features via extract_features_v7 (same as trainv7/hybrid_inference)
3. Fit CORAL adapter on source (training) vs target (unlabeled production) features
4. Transform source features to target domain (CORAL: source → target alignment)
5. Train XGBoost on CORAL-aligned features
6. Save: model + scaler + threshold + CORAL adapter + metadata

Usage (once relabeling done):
  python3 trainv8.py \
    --data-root /path/to/relabeled_eve_jsons \
    --target-eve /var/log/suricata/eve.json \
    --output ids_model_v8_final.pkl

REAL PATHS REQUIRED (marked with 🔴 REAL PATH REQUIRED):
"""

import gc, json, pickle, warnings, logging, argparse, random
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trainv8")

# ============================================================
# 🔴 REAL PATH REQUIRED: Base directory for relabeled eve_<Type>.json files
# Expected structure (output of relabel_cicids2018.py):
#   data_root/
#     eve_Benign.json
#     eve_DoS.json
#     eve_DDoS.json
#     eve_WebAttack.json
#     eve_Infiltration.json
#     eve_Bot.json
# ============================================================
DEFAULT_DATA_ROOT = Path("./data/relabeled_combined")  # 🔴 CHANGE ME

# 🔴 REAL PATH REQUIRED: Unlabeled production eve.json for CORAL target domain (50k+ flows)
DEFAULT_TARGET_EVE = Path("./data/eve/cicids2017_thursday_eve.json")  # 🔴 CHANGE ME

# Output model path
DEFAULT_OUT_MODEL = Path("./ids_model_v8_final.pkl")

# Class mapping (matches relabel_cicids2018.py output)
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

# CORAL settings
CORAL_LAMBDA_REG = 1e-5
CORAL_MAX_TARGET_SAMPLES = 50_000
CORAL_MAX_SOURCE_SAMPLES = 100_000  # reservoir sample from source for CORAL fit

FEATURE_DIM = 70  # extract_features_v7 output dimension
FEATURE_DTYPE = np.float32

# ============================================================
# RESERVOIR SAMPLING LIMITS (per class, max samples to keep)
# Memory budget: (300k + 5×150k) × 70 × 4 bytes ≈ 294 MB features
# Total with labels/overhead < 1 GB — well under 10 GB target
# ============================================================
MAX_SAMPLES_PER_CLASS = {
    "Benign": 300_000,
    "DoS": 150_000,
    "DDoS": 150_000,
    "WebAttack": 150_000,
    "Infiltration": 150_000,
    "Bot": 150_000,
}

def _extract_tcp_flags(event):
    tcp = event.get("tcp")
    if not tcp:
        return {"syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0}
    return {
        "syn": float(tcp.get("syn", 0) or 0),
        "ack": float(tcp.get("ack", 0) or 0),
        "fin": float(tcp.get("fin", 0) or 0),
        "rst": float(tcp.get("rst", 0) or 0),
        "psh": float(tcp.get("psh", 0) or 0),
    }


def extract_http_features(enrichment):
    if not enrichment or "http" not in enrichment:
        return {"mo": [0,0,0,0], "so": [0,0,0,0], "cth": 0, "cto": 0, "ctj": 0, "ct_": 0}
    http = enrichment["http"].get("http", {})
    method = (http.get("http_method") or "").upper()
    status = http.get("status", 0) or 0
    ctype  = (http.get("content_type") or "").lower()
    mo = [float(method=="GET"), float(method=="POST"), float(method=="HEAD"), float(method not in ("GET","POST","HEAD"))]
    so = [float(200<=status<300), float(300<=status<400), float(400<=status<500), float(500<=status<600)]
    return {"mo": mo, "so": so, "cth": float("html" in ctype), "cto": float("octet" in ctype),
            "ctj": float("json" in ctype), "ct_": float(ctype not in ("html","octet","json",""))}


def _extract_tls_features(enrichment):
    if not enrichment or "tls" not in enrichment:
        return {"e": 0, "sni": 0}
    tls = enrichment["tls"].get("tls", {})
    return {"e": 1, "sni": float(bool(tls.get("sni", "")))}

def _extract_dns_features(enrichment):
    if not enrichment or "dns" not in enrichment:
        return {"e": 0, "qa": 0, "qaa": 0, "qm": 0, "qo": 0, "rn": 0, "rnx": 0, "rr": 0, "ro": 0}
    dns = enrichment["dns"].get("dns", {})
    qt = set((q.get("rrtype") or "").upper() for q in (dns.get("queries") or []))
    rc = (dns.get("rcode") or "").upper()
    return {
        "e": 1,
        "qa": float("A" in qt), "qaa": float("AAAA" in qt),
        "qm": float("MX" in qt), "qo": float(bool(qt - {"A","AAAA","MX"})),
        "rn": float(rc == "NOERROR"), "rnx": float(rc == "NXDOMAIN"),
        "rr": float(rc == "REFUSED"), "ro": float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED")),
    }

def extract_features_v7(event, enrichment=None):
    """Extract 70 features from Suricata flow event (same as trainv7/hybrid_inference)."""
    if event.get("event_type") != "flow":
        return None
    from dateutil.parser import isoparse
    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)
    try:
        t0  = isoparse((flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1  = isoparse((flow.get("end") or "").replace("Z","+00:00"))
        dur = max((t1 - t0).total_seconds(), 0.0)
    except Exception:
        dur = 0.0

    dp  = int(event.get("dest_port", 0) or 0)
    sp  = int(event.get("src_port",  0) or 0)
    tp  = pts + ptc
    tb  = bts + btc
    sd  = max(dur, 0.1);  spk = max(tp, 1);  sb = max(tb, 1)

    proto     = (event.get("proto")     or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)
    state     = (flow.get("state")  or "").lower()
    reason    = (flow.get("reason") or "").lower()
    age       = float(flow.get("age", 0) or 0)

    base = np.array([
        dur, pts, ptc, bts, btc, tp, tb,
        tp/sd, tb/sd, tb/spk,
        bts/sb, pts/spk, abs(bts - btc)/sb,
        btc/sb, ptc/spk, age,
        float(ptc == 0),
        abs((bts / max(pts,1)) - (btc / max(ptc,1))),
        dp, sp,
        float(dp < 1024), float(1024 <= dp < 49152), float(dp >= 49152),
        float(sp == dp),
        float(ip_v == 6),
        float(proto == "TCP"), float(proto == "UDP"),
        float(proto in ("ICMP","ICMPv6")),
        float(app_proto == "http"), float(app_proto == "dns"),
        float(app_proto == "tls"),  float(app_proto == "dcerpc"),
        float(app_proto == "smb"),  float(app_proto == "rdp"),
        float(app_proto == "failed"),
        float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state == "established"), float(state == "closed"),
        float(state == "new"),
        float(reason == "timeout"), float(reason == "rst"),
        float(reason == "fin"),
    ], dtype=np.float32)

    enriched = np.zeros(28, dtype=np.float32)
    tcpf = _extract_tcp_flags(event)
    idx = 0
    enriched[idx:idx+5] = [tcpf["syn"], tcpf["ack"], tcpf["fin"], tcpf["rst"], tcpf["psh"]]
    idx += 5

    if enrichment:
        hf = extract_http_features(enrichment)
        enriched[idx:idx+4] = hf["mo"];  idx += 4
        enriched[idx:idx+4] = hf["so"];  idx += 4
        enriched[idx] = hf["cth"]; idx += 1
        enriched[idx] = hf["cto"]; idx += 1
        enriched[idx] = hf["ctj"]; idx += 1
        enriched[idx] = hf["ct_"]; idx += 1

        tlsf = _extract_tls_features(enrichment)
        enriched[idx] = tlsf["e"];   idx += 1
        enriched[idx] = tlsf["sni"]; idx += 1

        dnsf = _extract_dns_features(enrichment)
        enriched[idx] = dnsf["e"]; idx += 1
        enriched[idx:idx+4] = [dnsf["qa"], dnsf["qaa"], dnsf["qm"], dnsf["qo"]];  idx += 4
        enriched[idx:idx+4] = [dnsf["rn"], dnsf["rnx"], dnsf["rr"], dnsf["ro"]];  idx += 4

    return np.concatenate([base, enriched])

# ============================================================
# Data loading from relabeled eve_<Type>.json (recursive glob + reservoir sampling)
# ============================================================
def _reservoir_sample(stream, k: int, rng: random.Random):
    """Reservoir sampling: keep k items from a stream of unknown length."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def get_rss_memory_mb():
    """Returns current live RSS memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_labeled_data(data_root: Path, max_per_class: dict):
    """
    Recursively find all eve_<Class>.json files under data_root,
    stream JSON lines, extract 70 features on-the-fly, and populate
    pre-allocated numpy arrays using in-place reservoir sampling.

    Zero dict retention / Zero python list growth.
    """
    import glob

    # Collect all matching files per class
    class_files = {}
    for class_name in CLASS_MAP.keys():
        pattern = str(data_root / "**" / f"eve_{class_name}.json")
        files = glob.glob(pattern, recursive=True)
        if files:
            class_files[class_name] = sorted(files)
            log.info(f"  {class_name}: found {len(files)} file(s): {files}")
        else:
            log.warning(f"  {class_name}: no files found under {data_root}")

    if not class_files:
        raise FileNotFoundError(f"No eve_*.json files found under {data_root}")

    rng = random.Random(42)
    X_parts = []
    y_parts = []

    for class_name, files in class_files.items():
        label = CLASS_MAP[class_name]
        limit = max_per_class.get(class_name, 0)
        if limit <= 0:
            log.info(f"  {class_name}: limit=0, skipping")
            continue

        log.info(f"Streaming {class_name} from {len(files)} file(s) (limit={limit:,})...")
        
        # Sabit boyutlu float32 numpy array ayır
        X_class = np.empty((limit, FEATURE_DIM), dtype=FEATURE_DTYPE)
        n_seen = 0

        for fpath in files:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue

                    if event.get("event_type") != "flow":
                        continue

                    # Her flow'dan HEMEN extract_features_v7() ile öznitelik çıkar, dict'i ATIP unut
                    feat = extract_features_v7(event)
                    if feat is None:
                        continue

                    n_seen += 1

                    # In-place Reservoir Sampling
                    if n_seen <= limit:
                        X_class[n_seen - 1] = feat
                    else:
                        j = rng.randint(0, n_seen - 1)
                        if j < limit:
                            X_class[j] = feat

                    # Her 50k flow'da bir RSS bellek kullanımını logla
                    if n_seen % 50000 == 0:
                        rss_mb = get_rss_memory_mb()
                        log.info(
                            f"[MEMORY LOG] Class: {class_name:<12} | "
                            f"Flows Processed: {n_seen:>10,} | "
                            f"Samples Kept: {min(n_seen, limit):>7,} | "
                            f"RSS RAM: {rss_mb:7.2f} MB"
                        )

                    # Dosyanın geri kalanını okumadan hızlı çıkış için max_scan (limit * 2) sınırı
                    if n_seen >= limit * 2:
                        log.info(f"  Reached max scan limit ({n_seen:,} flows) for {class_name}, breaking early.")
                        break

            if n_seen >= limit * 2:
                break

        kept = min(n_seen, limit)
        if kept > 0:
            X_parts.append(X_class[:kept])
            y_parts.append(np.full(kept, label, dtype=np.int32))
            log.info(f"  -> {class_name}: {kept:,} flows sampled out of {n_seen:,} total (Final RSS: {get_rss_memory_mb():.2f} MB)")
        else:
            log.warning(f"  -> {class_name}: 0 valid flows extracted!")

        gc.collect()

    if not X_parts:
        raise ValueError("No features extracted from any class")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    log.info(f"Total source samples: {len(X):,} | Classes: {Counter(y)} | Final RSS RAM: {get_rss_memory_mb():.2f} MB")
    return X, y


def load_unlabeled_target(target_eve: Path, max_samples: int = 50_000):
    """Load unlabeled target domain features for CORAL (production traffic)."""
    X_target = np.empty((max_samples, FEATURE_DIM), dtype=np.float32)
    count = 0
    log.info(f"Loading target domain from {target_eve} (max {max_samples:,})...")
    with open(target_eve, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("event_type") != "flow":
                    continue
                feat = extract_features_v7(event)
                if feat is not None:
                    X_target[count] = feat
                    count += 1
                    if count % 10000 == 0:
                        log.info(f"[MEMORY LOG Target] Target Flows Loaded: {count:,} / {max_samples:,} | RSS RAM: {get_rss_memory_mb():.2f} MB")
            except Exception:
                continue

    log.info(f"  Target: {count:,} flows loaded (Final RSS: {get_rss_memory_mb():.2f} MB)")
    return X_target[:count]

# ============================================================
# CORAL Adapter (imported from coral_domain_adaptation.py)
# ============================================================
try:
    from coral_domain_adaptation import CORALDomainAdapter
except ImportError:
    # Fallback: inline minimal CORALDomainAdapter (should match coral_domain_adaptation.py)
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_array, check_is_fitted
    import numpy as np
    import pickle

    class CORAL(BaseEstimator, TransformerMixin):
        def __init__(self, lambda_reg=1e-5, copy=True):
            self.lambda_reg = lambda_reg
            self.copy = copy
            self.Cs_ = None
            self.Ct_ = None
            self.transform_matrix_ = None
            self.source_mean_ = None
            self.target_mean_ = None
            self.fitted_ = False

        def _compute_covariance(self, X):
            n_samples, n_features = X.shape
            cov = np.cov(X.T) + self.lambda_reg * np.eye(n_features)
            return cov

        def _matrix_sqrt_inv(self, C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        def _matrix_sqrt(self, C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        def fit(self, Xs, Xt):
            Xs = check_array(Xs, dtype=np.float64, copy=self.copy)
            Xt = check_array(Xt, dtype=np.float64, copy=self.copy)
            if Xs.shape[1] != Xt.shape[1]:
                raise ValueError(f"Feature dimension mismatch: source {Xs.shape[1]} vs target {Xt.shape[1]}")
            self.source_mean_ = Xs.mean(axis=0)
            self.target_mean_ = Xt.mean(axis=0)
            Xs_centered = Xs - self.source_mean_
            Xt_centered = Xt - self.target_mean_
            self.Cs_ = self._compute_covariance(Xs_centered)
            self.Ct_ = self._compute_covariance(Xt_centered)
            Cs_sqrt_inv = self._matrix_sqrt_inv(self.Cs_)
            Ct_sqrt = self._matrix_sqrt(self.Ct_)
            self.transform_matrix_ = Cs_sqrt_inv @ Ct_sqrt
            self.fitted_ = True
            return self

        def transform(self, X, domain='source'):
            check_is_fitted(self, 'fitted_')
            X = check_array(X, dtype=np.float64, copy=self.copy)
            if domain == 'source':
                X_centered = X - self.source_mean_
                X_aligned = X_centered @ self.transform_matrix_
                X_aligned = X_aligned + self.target_mean_
            elif domain == 'target':
                inv_transform = np.linalg.inv(self.transform_matrix_)
                X_centered = X - self.target_mean_
                X_aligned = X_centered @ inv_transform
                X_aligned = X_aligned + self.source_mean_
            else:
                raise ValueError("domain must be 'source' or 'target'")
            return X_aligned.astype(np.float32)

        def get_alignment_quality(self):
            check_is_fitted(self, 'fitted_')
            cov_diff = self.Cs_ - self.Ct_
            frobenius_norm = float(np.linalg.norm(cov_diff, 'fro'))
            trace_ratio = float(np.trace(self.Ct_) / np.trace(self.Cs_))
            return {
                'frobenius_distance': frobenius_norm,
                'trace_ratio_target_over_source': trace_ratio,
                'condition_number_source': float(np.linalg.cond(self.Cs_)),
                'condition_number_target': float(np.linalg.cond(self.Ct_)),
            }

    class CORALDomainAdapter(BaseEstimator, TransformerMixin):
        def __init__(self, lambda_reg=1e-5, scaler=None):
            self.lambda_reg = lambda_reg
            self.coral = CORAL(lambda_reg=lambda_reg)
            self.scaler = scaler or StandardScaler()
            self.is_fitted_ = False
            self.n_features_ = None
            self.alignment_metrics_ = {}

        def fit(self, X_source, X_target, scale=True):
            X_source = check_array(X_source, dtype=np.float64)
            X_target = check_array(X_target, dtype=np.float64)
            self.n_features_ = X_source.shape[1]
            if X_target.shape[1] != self.n_features_:
                raise ValueError(f"Feature mismatch: source={self.n_features_}, target={X_target.shape[1]}")
            if scale:
                X_source_scaled = self.scaler.fit_transform(X_source)
                X_target_scaled = self.scaler.transform(X_target)
            else:
                X_source_scaled = X_source
                X_target_scaled = X_target
            self.coral.fit(X_source_scaled, X_target_scaled)
            self.is_fitted_ = True
            self.alignment_metrics_ = self.coral.get_alignment_quality()
            return self

        def transform_source(self, X_source):
            if not self.is_fitted_: raise RuntimeError("Not fitted")
            X_scaled = self.scaler.transform(X_source)
            return self.coral.transform(X_scaled, domain='source')

        def transform_target(self, X_target):
            if not self.is_fitted_: raise RuntimeError("Not fitted")
            X_scaled = self.scaler.transform(X_target)
            return self.coral.transform(X_scaled, domain='target')

        def transform(self, X, domain='target'):
            if domain == 'source': return self.transform_source(X)
            elif domain == 'target': return self.transform_target(X)
            else: raise ValueError("domain must be 'source' or 'target'")

        def get_metrics(self):
            if not self.is_fitted_: return {}
            return {**self.alignment_metrics_, 'n_features': self.n_features_, 'lambda_reg': self.lambda_reg}

        def save(self, path):
            if not self.is_fitted_: raise RuntimeError("Cannot save unfitted adapter.")
            with open(path, 'wb') as f:
                pickle.dump({'coral': self.coral, 'scaler': self.scaler, 'is_fitted_': self.is_fitted_,
                             'n_features_': self.n_features_, 'alignment_metrics_': self.alignment_metrics_}, f)

        @classmethod
        def load(cls, path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            obj = cls()
            obj.coral, obj.scaler, obj.is_fitted_, obj.n_features_, obj.alignment_metrics_ = \
                d['coral'], d['scaler'], d['is_fitted_'], d['n_features_'], d['alignment_metrics_']
            return obj

# ============================================================
# Threshold optimization (same as trainv7)
# ============================================================
def find_best_threshold(y_true, y_prob, min_recall=0.95):
    """Find threshold achieving min_recall with lowest FAR."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    valid = np.where(rec >= min_recall)[0]
    if len(valid) == 0:
        return 0.5
    best_idx = valid[np.argmin(1 - prec[valid])]
    return float(thr[best_idx]) if best_idx < len(thr) else 1.0


# ============================================================
# Main training pipeline
# ============================================================
def main(args):
    log.info("=" * 60)
    log.info("TRAINV8 — XGBoost + CORAL Domain Adaptation")
    log.info("=" * 60)

    # 1. Load labeled source data
    log.info("\n[1/6] Loading labeled source data (relabeled eve_<Type>.json)...")
    X_source, y_source = load_labeled_data(args.data_root, MAX_SAMPLES_PER_CLASS)
    log.info(f"Total source samples: {len(X_source)} | Classes: {Counter(y_source)}")

    # 2. Load unlabeled target data for CORAL
    log.info("\n[2/6] Loading unlabeled target data for CORAL...")
    X_target = load_unlabeled_target(args.target_eve, CORAL_MAX_TARGET_SAMPLES)

    # 3. Fit CORAL adapter (source -> target alignment for training)
    log.info("\n[3/6] Fitting CORAL adapter (source→target)...")
    coral = CORALDomainAdapter(lambda_reg=CORAL_LAMBDA_REG)
    coral.fit(X_source, X_target, scale=True)
    metrics = coral.get_metrics()
    log.info(f"  Frobenius distance: {metrics.get('frobenius_distance', 'N/A'):.4f}")
    log.info(f"  Trace ratio (target/source): {metrics.get('trace_ratio_target_over_source', 'N/A'):.4f}")
    log.info(f"  Condition numbers: src={metrics.get('condition_number_source', 'N/A'):.2f}, tgt={metrics.get('condition_number_target', 'N/A'):.2f}")

    # 4. Transform source features to target domain (CORAL alignment)
    log.info("\n[4/6] Applying CORAL alignment to source features...")
    X_aligned = coral.transform_source(X_source)

    # 5. Train/val split (stratified)
    log.info("\n[5/6] Train/val split & XGBoost training...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_aligned, y_source, test_size=0.2, random_state=42, stratify=y_source
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Class weights
    sample_weight = compute_sample_weight('balanced', y_train)

    # XGBoost params (same as trainv7)
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

    dtrain = xgb.DMatrix(X_train_s, label=y_train, weight=sample_weight)
    dval = xgb.DMatrix(X_val_s, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params, dtrain, num_boost_round=2000, evals=evals,
                    early_stopping_rounds=50, verbose_eval=50)

    # 6. Threshold optimization on validation (binary: Benign vs Attack)
    log.info("\n[6/6] Optimizing binary threshold (Benign vs Attack)...")
    val_probs = bst.predict(dval)
    y_val_bin = (y_val != 0).astype(int)
    prob_atk = 1.0 - val_probs[:, 0]
    best_thr = find_best_threshold(y_val_bin, prob_atk, min_recall=0.95)
    log.info(f"  Best threshold (recall>=0.95): {best_thr:.4f}")

    # Final metrics
    val_preds = (prob_atk >= best_thr).astype(int)
    log.info(f"\nValidation Binary Classification:")
    log.info(f"  FAR: {(val_preds[y_val_bin==0]==1).mean():.4f}")
    log.info(f"  Recall: {(val_preds[y_val_bin==1]==1).mean():.4f}")
    log.info(f"\nMulticlass Report:")
    log.info(classification_report(y_val, val_probs.argmax(axis=1), target_names=list(INV_CLASS_MAP.values())))

    # Save bundle
    bundle = {
        'model': bst,
        'scaler': scaler,
        'threshold': best_thr,
        'class_map': CLASS_MAP,
        'inv_class_map': INV_CLASS_MAP,
        'coral_adapter': coral,  # For inference: transform_target()
        'coral_metrics': metrics,
        'feature_dim': X_source.shape[1],
        'train_date': datetime.now().isoformat(),
        'num_classes': NUM_CLASSES,
        'xgb_params': params,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"\n✅ Model saved to {args.output}")
    log.info(f"   Threshold: {best_thr:.4f}")
    log.info(f"   CORAL adapter included for inference pipeline")
    log.info(f"   Use coral_adapter.transform_target() on live features before scaler")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainv8 — XGBoost + CORAL Domain Adaptation")
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT,
                        help='🔴 Directory containing relabeled eve_<Type>.json files')
    parser.add_argument('--target-eve', type=Path, default=DEFAULT_TARGET_EVE,
                        help='🔴 Unlabeled production eve.json for CORAL target domain')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT_MODEL,
                        help='Output model path')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU for XGBoost')
    args = parser.parse_args()
    main(args)

# ============================================================
# 🔴 REAL PATHS REQUIRED — UPDATE BEFORE RUNNING:
# ============================================================
# 1. DEFAULT_DATA_ROOT: Path to relabeled eve_<Type>.json files
#    (output of relabel_cicids2018.py)
#    Expected: eve_Benign.json, eve_DoS.json, eve_DDoS.json,
#              eve_WebAttack.json, eve_Infiltration.json, eve_Bot.json
#
# 2. DEFAULT_TARGET_EVE: Unlabeled production eve.json (Suricata live)
#    Used for CORAL target covariance estimation (50k+ flows)
#    Typical: /var/log/suricata/eve.json
#
# 3. DEFAULT_OUT_MODEL: Where to save ids_model_v8_final.pkl
#
# After relabeling completes, just run:
#   python3 trainv8.py --data-root /path/to/relabeled --target-eve /var/log/suricata/eve.json