# -*- coding: utf-8 -*-
"""
trainv7.py — XGBoost IDS with enriched features (tcp-flags, http, tls, dns)
==========================================================================
Builds on v6: adds ~20 new features extracted from Suricata's tcp, http, tls,
and dns event types, correlated by flow_id.

Pipeline:
  1. Single-pass EVE JSON reading: cache http/tls/dns per flow_id
  2. When a flow event arrives, enrich feature vector with cached data
  3. Train XGBoost with same hyperparams as v6 (but more features)

Output: ids_model_v7_final.pkl
"""

import gc, json, pickle, warnings, logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

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
log = logging.getLogger("trainv7")

BASE_DIR  = Path(__file__).resolve().parent.parent / "data"
OUT_MODEL = Path(__file__).resolve().parent.parent / "models" / "archive" / "ids_model_v7_final.pkl"

MAX_BENIGN_PER_FILE        = 300_000
MAX_PER_ATTACK_CLASS_TOTAL = 500_000

EVE_FILES = {
    "Benign": [
        BASE_DIR / "Wednesday-14-02-2018/eve_Benign.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407021400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407071400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408111400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408121400.json",
    ],
    "DoS": [
        BASE_DIR / "Thursday-15-02-2018/eve_DoS.json",
        BASE_DIR / "Friday-16-02-2018/eve_DoS.json",
    ],
    "DDoS": [
        BASE_DIR / "Tuesday-20-02-2018/eve_DDoS.json",
        BASE_DIR / "Wednesday-21-02-2018/eve_DDoS.json",
    ],
    "WebAttack": [
        BASE_DIR / "Thursday-22-02-2018/eve_WebAttack.json",
        BASE_DIR / "Friday-23-02-2018/eve_WebAttack.json",
    ],
    "Infiltration": [
        BASE_DIR / "Wednesday-28-02-2018/eve_Infiltration.json",
        BASE_DIR / "Thursday-01-03-2018/eve_Infiltration.json",
    ],
    "Bot": [
        BASE_DIR / "Friday-02-03-2018/eve_Bot.json",
        BASE_DIR / "Ctu-13/eve_botnet_neris.json",
        BASE_DIR / "mcfp felk/eve_botnet_mcfp.json",
    ],
}

CLASS_MAP     = {"Benign":0, "DoS":1, "DDoS":2, "WebAttack":3, "Infiltration":4, "Bot":5}
INV_CLASS_MAP = {v:k for k,v in CLASS_MAP.items()}
N_CLASSES     = len(CLASS_MAP)

FEATURE_COLS_v6 = [
    "duration_s",
    "pkts_toserver", "pkts_toclient",
    "bytes_toserver", "bytes_toclient",
    "total_pkts", "total_bytes",
    "pkt_rate", "byte_rate", "bytes_per_pkt",
    "upload_byte_ratio", "upload_pkt_ratio", "byte_asymmetry",
    "download_byte_ratio", "download_pkt_ratio",
    "flow_age",
    "is_unidirectional",
    "pkt_size_variance_proxy",
    "dest_port", "src_port",
    "is_well_known_port", "is_registered_port", "is_high_port",
    "is_same_port",
    "is_ipv6", "is_tcp", "is_udp", "is_icmp",
    "app_http", "app_dns", "app_tls",
    "app_dcerpc", "app_smb", "app_rdp",
    "app_failed", "app_unknown",
    "state_established", "state_closed", "state_new",
    "reason_timeout", "reason_rst", "reason_fin",
]

FEATURE_COLS_v7 = FEATURE_COLS_v6 + [
    "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst", "tcp_psh",
    "http_method_get", "http_method_post", "http_method_head", "http_method_other",
    "http_status_2xx", "http_status_3xx", "http_status_4xx", "http_status_5xx",
    "http_content_type_html", "http_content_type_octet", "http_content_type_json",
    "http_content_type_other",
    "tls_exists", "tls_sni_exists",
    "dns_exists", "dns_query_a", "dns_query_aaaa", "dns_query_mx", "dns_query_other",
    "dns_rcode_noerror", "dns_rcode_nxdomain", "dns_rcode_refused", "dns_rcode_other",
]

N_FEATURES_v7 = len(FEATURE_COLS_v7)


class FlowEnrichment:
    """Single-pass cache for http/tls/dns events keyed by flow_id."""

    def __init__(self, max_entries=500_000):
        self._store = {}
        self._max = max_entries

    def ingest(self, event: dict):
        etype = event.get("event_type")
        if etype not in ("http", "tls", "dns"):
            return
        flow_id = event.get("flow_id")
        if flow_id is None:
            return
        if flow_id in self._store and etype in self._store[flow_id]:
            return
        if len(self._store) >= self._max:
            return
        entry = self._store.setdefault(flow_id, {})
        entry["etype"] = etype
        entry[etype] = event

    def get(self, flow_id):
        return self._store.pop(flow_id, None)

    def clear(self):
        self._store.clear()


def extract_tcp_flags(flow_event: dict) -> dict:
    tcp = flow_event.get("tcp") if flow_event else None
    if not tcp:
        return {"syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0}
    flags_str = tcp.get("tcp_flags_ts", "00")
    try:
        flags_int = int(flags_str, 16)
    except (ValueError, TypeError):
        flags_int = 0
    return {
        "syn": float(bool(flags_int & 0x02)),
        "ack": float(bool(flags_int & 0x10)),
        "fin": float(bool(flags_int & 0x01)),
        "rst": float(bool(flags_int & 0x04)),
        "psh": float(bool(flags_int & 0x08)),
    }


def extract_http_features(http_event: dict) -> dict:
    if not http_event:
        return {"method_onehot": [0,0,0,0], "status_onehot": [0,0,0,0],
                "ct_html": 0, "ct_octet": 0, "ct_json": 0, "ct_other": 0}
    http = http_event.get("http", {})
    method = (http.get("http_method") or "").upper()
    method_vec = [0,0,0,0]
    if method == "GET":    method_vec[0] = 1
    elif method == "POST":   method_vec[1] = 1
    elif method == "HEAD":   method_vec[2] = 1
    else:                    method_vec[3] = 1 if method else 0
    status = int(http.get("status", 0) or 0)
    status_vec = [0,0,0,0]
    if 200 <= status < 300:   status_vec[0] = 1
    elif 300 <= status < 400: status_vec[1] = 1
    elif 400 <= status < 500: status_vec[2] = 1
    elif status >= 500:        status_vec[3] = 1
    ct = (http.get("http_content_type") or "").lower()
    ct_html = float("html" in ct)
    ct_octet = float("octet-stream" in ct or "binary" in ct)
    ct_json = float("json" in ct)
    ct_other = float(not (ct_html or ct_octet or ct_json) and bool(ct))
    return {"method_onehot": method_vec, "status_onehot": status_vec,
            "ct_html": ct_html, "ct_octet": ct_octet, "ct_json": ct_json, "ct_other": ct_other}


def extract_tls_features(tls_event: dict) -> dict:
    if not tls_event:
        return {"exists": 0, "sni_exists": 0}
    tls = tls_event.get("tls", {})
    sni = tls.get("sni") or ""
    return {"exists": 1, "sni_exists": float(bool(sni))}


def extract_dns_features(dns_event: dict) -> dict:
    if not dns_event:
        return {"exists": 0, "q_a": 0, "q_aaaa": 0, "q_mx": 0, "q_other": 0,
                "r_noerror": 0, "r_nxdomain": 0, "r_refused": 0, "r_other": 0}
    dns = dns_event.get("dns", {})
    queries = dns.get("queries") or []
    qtypes = set()
    for q in queries:
        rrtype = (q.get("rrtype") or "").upper()
        if rrtype:
            qtypes.add(rrtype)
    q_a = float("A" in qtypes)
    q_aaaa = float("AAAA" in qtypes)
    q_mx = float("MX" in qtypes)
    q_other = float(bool(qtypes - {"A","AAAA","MX"}))
    rcode = (dns.get("rcode") or "").upper()
    r_noerror = float(rcode == "NOERROR")
    r_nxdomain = float(rcode == "NXDOMAIN")
    r_refused = float(rcode == "REFUSED")
    r_other = float(bool(rcode) and rcode not in ("NOERROR","NXDOMAIN","REFUSED"))
    return {"exists": 1, "q_a": q_a, "q_aaaa": q_aaaa, "q_mx": q_mx, "q_other": q_other,
            "r_noerror": r_noerror, "r_nxdomain": r_nxdomain, "r_refused": r_refused, "r_other": r_other}


def extract_features_v7(event: dict, enrichment: dict) -> np.ndarray:
    """Extract v7 features (42 base + ~25 enriched) from a flow event + enrichment dict."""
    if event.get("event_type") != "flow":
        return None
    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)
    try:
        from dateutil.parser import isoparse
        t0 = isoparse((flow.get("start") or event.get("timestamp", "")).replace("Z", "+00:00"))
        t1 = isoparse((flow.get("end") or "").replace("Z", "+00:00"))
        dur = max((t1 - t0).total_seconds(), 0.0)
    except Exception:
        dur = 0.0
    dp  = int(event.get("dest_port", 0) or 0)
    sp  = int(event.get("src_port",  0) or 0)
    tp  = pts + ptc
    tb  = bts + btc
    MIN_DUR = 0.1; sd = max(dur, MIN_DUR); spk = max(tp, 1); sb = max(tb, 1)
    proto     = (event.get("proto") or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)
    state     = (flow.get("state")  or "").lower()
    reason    = (flow.get("reason") or "").lower()
    age       = float(flow.get("age", 0) or 0)
    base = np.array([
        dur, pts, ptc, bts, btc, tp, tb,
        tp/sd, tb/sd, tb/spk,
        bts/sb, pts/spk, abs(bts-btc)/sb,
        btc/sb, ptc/spk, age,
        float(ptc==0),
        abs((bts/max(pts,1)) - (btc/max(ptc,1))),
        dp, sp,
        float(dp<1024), float(1024<=dp<49152), float(dp>=49152), float(sp==dp),
        float(ip_v==6), float(proto=="TCP"), float(proto=="UDP"), float(proto in ("ICMP","ICMPv6")),
        float(app_proto=="http"), float(app_proto=="dns"), float(app_proto=="tls"),
        float(app_proto=="dcerpc"), float(app_proto=="smb"), float(app_proto=="rdp"),
        float(app_proto=="failed"), float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state=="established"), float(state=="closed"), float(state=="new"),
        float(reason=="timeout"), float(reason=="rst"), float(reason=="fin"),
    ], dtype=np.float32)
    enriched = np.zeros(N_FEATURES_v7 - len(base), dtype=np.float32)
    tf = extract_tcp_flags(event)
    idx = 0
    enriched[idx:idx+5] = [tf["syn"], tf["ack"], tf["fin"], tf["rst"], tf["psh"]]; idx+=5
    if enrichment:
        hf = extract_http_features(enrichment)
        enriched[idx:idx+4] = hf["method_onehot"]; idx+=4
        enriched[idx:idx+4] = hf["status_onehot"]; idx+=4
        enriched[idx] = hf["ct_html"]; idx+=1
        enriched[idx] = hf["ct_octet"]; idx+=1
        enriched[idx] = hf["ct_json"]; idx+=1
        enriched[idx] = hf["ct_other"]; idx+=1
        tlsf = extract_tls_features(enrichment)
        enriched[idx] = tlsf["exists"]; idx+=1
        enriched[idx] = tlsf["sni_exists"]; idx+=1
        dnsf = extract_dns_features(enrichment)
        enriched[idx] = dnsf["exists"]; idx+=1
        enriched[idx:idx+4] = [dnsf["q_a"], dnsf["q_aaaa"], dnsf["q_mx"], dnsf["q_other"]]; idx+=4
        enriched[idx:idx+4] = [dnsf["r_noerror"], dnsf["r_nxdomain"], dnsf["r_refused"], dnsf["r_other"]]; idx+=4
    return np.concatenate([base, enriched])


def load_eve_file_v7(path: Path, label_id: int, max_samples: int):
    """Single-pass EVE reader with flow_id-based enrichment caching."""
    if not path.exists():
        log.warning(f"  dosya yok: {path.name}")
        return None, None
    cache = FlowEnrichment(max_entries=max_samples * 2)
    X, count = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_samples is not None and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            etype = ev.get("event_type")
            if etype == "flow":
                enrichment = cache.get(ev.get("flow_id"))
                feat = extract_features_v7(ev, enrichment)
                if feat is not None:
                    X.append(feat)
                    count += 1
            elif etype in ("http", "tls", "dns"):
                cache.ingest(ev)
    if not X:
        return None, None
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.full(len(X_arr), label_id, dtype=np.int8)
    log.info(f"    OK  {path.name:<55} -> {len(X_arr):>10,} ornek")
    return X_arr, y_arr


def load_all_data_v7():
    all_X, all_y = [], []
    for label_name, paths in EVE_FILES.items():
        label_id = CLASS_MAP[label_name]
        file_list = paths if isinstance(paths, list) else [paths]
        if label_name == "Benign":
            per_file_limit = MAX_BENIGN_PER_FILE
        else:
            per_file_limit = max(1, MAX_PER_ATTACK_CLASS_TOTAL // len(file_list))
        class_X, class_y = [], []
        log.info(f"[{label_name}] {len(file_list)} dosya, dosya basina <= {per_file_limit:,}")
        for p in file_list:
            X, y = load_eve_file_v7(p, label_id, per_file_limit)
            if X is not None:
                class_X.append(X)
                class_y.append(y)
            del X, y; gc.collect()
        if class_X:
            cat_X = np.concatenate(class_X)
            cat_y = np.concatenate(class_y)
            if label_name != "Benign" and len(cat_X) > MAX_PER_ATTACK_CLASS_TOTAL:
                idx = np.random.choice(len(cat_X), MAX_PER_ATTACK_CLASS_TOTAL, replace=False)
                cat_X = cat_X[idx]; cat_y = cat_y[idx]
            all_X.append(cat_X); all_y.append(cat_y)
            log.info(f"  -> {label_name:<14} toplam: {len(cat_X):>10,}")
            del class_X, class_y, cat_X, cat_y; gc.collect()
    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    del all_X, all_y
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"\n  > Toplam ham veri: {len(y_all):,} akis")
    return X_all, y_all


def find_best_threshold(y_true_bin, y_prob_atk):
    best_t, best_rec = 0.5, 0.0
    for t in np.arange(0.50, 1.00, 0.01):
        pred = (y_prob_atk >= t).astype(int)
        tn = ((pred==0)&(y_true_bin==0)).sum()
        fp = ((pred==1)&(y_true_bin==0)).sum()
        fn = ((pred==0)&(y_true_bin!=0)).sum()
        tp = ((pred==1)&(y_true_bin!=0)).sum()
        far = fp / (fp+tn+1e-10)
        rec = tp / (tp+fn+1e-10)
        if far < 0.01 and rec > best_rec:
            best_rec = rec; best_t = t
    return round(float(best_t), 2), round(float(best_rec), 4)


def train():
    print("\n" + "="*64)
    print("   IDS v7 - ZENGIN OZELLIKLI XGBoost MODELI")
    print("="*64)
    print(f"  Ozellik sayisi : {N_FEATURES_v7} ({len(FEATURE_COLS_v6)} base + {N_FEATURES_v7 - len(FEATURE_COLS_v6)} enriched)")
    print(f"  Siniflar       : {list(CLASS_MAP.keys())}")

    log.info("Veri yukleniyor (v7 enriched)...")
    X_all, y_all = load_all_data_v7(); gc.collect()
    cnt = Counter(y_all.tolist())
    log.info("Sinif dagilimi:")
    for cid in sorted(cnt.keys()):
        log.info(f"  {INV_CLASS_MAP[cid]:<14}: {cnt[cid]:>12,}  ({cnt[cid]/len(y_all)*100:5.2f}%)")

    log.info("Stratified train/val/test split (70/15/15)...")
    X_tr, X_rest, y_tr, y_rest = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=42)
    del X_all, y_all, X_rest, y_rest; gc.collect()
    log.info(f"  Train: {len(y_tr):>10,}   Val: {len(y_val):>9,}   Test: {len(y_te):>9,}")

    log.info("StandardScaler fit ediliyor...")
    sample = np.random.choice(len(y_tr), min(300_000, len(y_tr)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_tr[sample]); del sample
    X_tr  = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32); gc.collect()

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    log.info(f"  Sinif agirliklari: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    log.info("XGBoost modeli olusturuluyor...")
    model = xgb.XGBClassifier(
        n_estimators=1500, max_depth=9, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.85,
        min_child_weight=2, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist", device="cuda",
        max_bin=256, early_stopping_rounds=30,
        random_state=42, n_jobs=-1, verbosity=1,
    )
    log.info("Egitim basliyor...")
    t0 = datetime.now()
    model.fit(X_tr, y_tr, sample_weight=sample_weights,
              eval_set=[(X_val, y_val), (X_te, y_te)], verbose=100)
    train_dur = (datetime.now()-t0).total_seconds()
    log.info(f"Egitim tamamlandi: {train_dur/60:.1f} dk, best_iter={model.best_iteration}")

    log.info("Test seti degerlendiriliyor...")
    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)
    y_te_bin   = (y_te != 0).astype(int)
    y_prob_atk = 1.0 - y_prob[:, 0]
    acc        = accuracy_score(y_te, y_pred)
    macro_f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te_bin, (y_pred != 0).astype(int)).ravel()
    far = fp / (fp+tn+1e-10); rec = tp / (tp+fn+1e-10)
    p, r, _ = precision_recall_curve(y_te_bin, y_prob_atk)
    pr_auc = auc(r, p)
    best_t, best_rec = find_best_threshold(y_te_bin, y_prob_atk)
    y_tuned = (y_prob_atk >= best_t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_te_bin, y_tuned).ravel()
    far2 = fp2 / (fp2+tn2+1e-10); rec2 = tp2 / (tp2+fn2+1e-10)

    print("\n" + "="*64)
    print("   v7 TEST SONUCLARI")
    print("="*64)
    print(f"  Sure             : {train_dur/60:.1f} dk")
    print(f"  Best iteration   : {model.best_iteration}")
    print(f"  >> Multiclass Acc: {acc*100:6.2f}%")
    print(f"  >> Macro F1      : {macro_f1*100:6.2f}%")
    print(f"  >> Weighted F1   : {weighted_f1*100:6.2f}%")
    print(f"  >> Binary FAR    : {far*100:6.3f}%")
    print(f"  >> Binary Recall : {rec*100:6.2f}%")
    print(f"  >> PR-AUC        : {pr_auc:.4f}")
    print(f"  >> Opt FAR (t={best_t}): {far2*100:6.3f}%")
    print(f"  >> Opt Recall    : {rec2*100:6.2f}%")

    present_labels = sorted(np.unique(y_te).tolist())
    present_names  = [INV_CLASS_MAP[i] for i in present_labels]
    print(f"\n  --- Sinif Bazli Rapor ---")
    print(classification_report(y_te, y_pred, labels=present_labels, target_names=present_names, zero_division=0))
    print("  --- Karisiklik Matrisi ---")
    cm = confusion_matrix(y_te, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)
    print(cm_df.to_string())

    print("\n  --- En Onemli 20 Ozellik ---")
    fi = sorted(zip(FEATURE_COLS_v7, model.feature_importances_), key=lambda x: -x[1])[:20]
    for fname, imp in fi:
        print(f"    {fname:<35} {imp:.4f}  {'#'*int(imp*100)}")

    bundle = {
        "model": model, "scaler": scaler,
        "feature_cols": FEATURE_COLS_v7,
        "class_map": CLASS_MAP, "inv_class_map": INV_CLASS_MAP,
        "threshold": best_t,
        "metrics": dict(accuracy=acc, far=far2, recall=rec2, pr_auc=pr_auc,
                        macro_f1=macro_f1, weighted_f1=weighted_f1),
        "trained_on": "CICIDS2018 + MAWI + CTU-13 + MCFP (v7 enriched)",
        "n_features": N_FEATURES_v7,
        "n_features_base": len(FEATURE_COLS_v6),
        "best_iteration": int(model.best_iteration),
        "training_duration_min": round(train_dur/60, 2),
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Model kaydedildi: {OUT_MODEL}")
    log.info(f"  Threshold  : {best_t}")
    log.info(f"  Features   : {N_FEATURES_v7} ({len(FEATURE_COLS_v6)} + {N_FEATURES_v7 - len(FEATURE_COLS_v6)})")
    log.info(f"  Macro F1   : {macro_f1*100:.2f}%")
    print("\n  EGITIM TAMAMLANDI")
    print("="*64)


if __name__ == "__main__":
    train()
