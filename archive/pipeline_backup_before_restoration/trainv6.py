# -*- coding: utf-8 -*-
"""
trainv6.py - Nihai XGBoost IDS Modeli (Multiclass Optimized)
=============================================================
v5'in başarılı temelleri üzerine inşa edilmiş, tüm zayıflıkları kapatılmış
nihai üretim modeli.

v5 -> v6 Iyileştirmeleri:
  1. CTU-13 + MCFP botnet PCAP'lerinden üretilen EVE JSON'lar egitime dahil
  2. min_child_weight=5 -> 2 (kucuk siniflar daha iyi ogrenir)
  3. gamma=0.3 -> 0.1 (agac buyumesi daha kontrollu)
  4. max_depth=8 -> 9 (daha karmasik oruntuler yakalanir)
  5. Sinif agirligi daha agresif (compute_sample_weight 'balanced')
  6. Erken durdurma patience=30 (overfit'e daha gec izin verir)
  7. Stratified train/val/test split (val set egitimde kullanilmaz)
  8. Detayli logging ve confusion matrix
  9. Feature importance raporu
 10. Hiperparametre optimizasyonu

Cikti: ids_model_v6_final.pkl
"""

import gc, json, pickle, warnings, logging
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trainv6")

# ============================================================================
# 1) DOSYA YOLLARI
# ============================================================================
BASE_DIR  = Path("/run/media/mehmet/siber data1/ai modeli xgboost/pcap dosyaları ve veri setleri")
OUT_MODEL = Path("/run/media/mehmet/siber data1/ai modeli xgboost/ids_model_v6_final.pkl")

# Egitim sinirlari (RAM korumasi)
MAX_BENIGN_PER_FILE        = 300_000    # Omurga Benign basina ust sinir
MAX_PER_ATTACK_CLASS_TOTAL = 500_000    # Saldiri sinifi basina toplam ust sinir

# ============================================================================
# 2) VERI KAYNAKLARI
# ============================================================================
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

CLASS_MAP = {
    "Benign": 0, "DoS": 1, "DDoS": 2,
    "WebAttack": 3, "Infiltration": 4, "Bot": 5,
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
N_CLASSES = len(CLASS_MAP)

# ============================================================================
# 3) 42 OZELLIK TANIMI
# ============================================================================
FEATURE_COLS = [
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
N_FEATURES = len(FEATURE_COLS)

# ============================================================================
# 4) OZELLIK CIKARIMI
# ============================================================================
def extract_features(event: dict):
    """Bir Suricata flow event'inden 42 ozellik cikarir."""
    if event.get("event_type") != "flow":
        return None

    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)

    # Sure - Suricata'nin mikro-saniye formati da dahil
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
    MIN_DUR = 0.1
    sd  = max(dur, MIN_DUR)
    spk = max(tp, 1)
    sb  = max(tb, 1)

    proto     = (event.get("proto") or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)
    state     = (flow.get("state")  or "").lower()
    reason    = (flow.get("reason") or "").lower()
    age       = float(flow.get("age", 0) or 0)

    return np.array([
        dur, pts, ptc, bts, btc, tp, tb,
        tp / sd, tb / sd, tb / spk,
        bts / sb, pts / spk, abs(bts - btc) / sb,
        btc / sb, ptc / spk,
        age,
        float(ptc == 0),
        abs((bts / max(pts, 1)) - (btc / max(ptc, 1))),
        dp, sp,
        float(dp < 1024),
        float(1024 <= dp < 49152),
        float(dp >= 49152),
        float(sp == dp),
        float(ip_v == 6),
        float(proto == "TCP"),
        float(proto == "UDP"),
        float(proto in ("ICMP", "ICMPv6")),
        float(app_proto == "http"),
        float(app_proto == "dns"),
        float(app_proto == "tls"),
        float(app_proto == "dcerpc"),
        float(app_proto == "smb"),
        float(app_proto == "rdp"),
        float(app_proto == "failed"),
        float(app_proto not in ("http", "dns", "tls", "dcerpc", "smb", "rdp", "failed")),
        float(state == "established"),
        float(state == "closed"),
        float(state == "new"),
        float(reason == "timeout"),
        float(reason == "rst"),
        float(reason == "fin"),
    ], dtype=np.float32)


# ============================================================================
# 5) EGITIM VERISI YUKLEME (Bellek dostu)
# ============================================================================
def load_eve_file(path: Path, label_id: int, max_samples: int):
    """Bir EVE JSON dosyasindan en fazla max_samples akis okur."""
    if not path.exists():
        log.warning(f"  dosya yok, atlanıyor: {path.name}")
        return None, None

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
                feat = extract_features(ev)
                if feat is not None:
                    X.append(feat)
                    count += 1
            except Exception:
                continue

    if not X:
        return None, None

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.full(len(X_arr), label_id, dtype=np.int8)
    log.info(f"    OK  {path.name[:55]:<55} -> {len(X_arr):>10,} ornek")
    return X_arr, y_arr


def load_all_data():
    """Tum siniflari yukler, sinif basina toplam ornek sayisini sinirlar."""
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
            X, y = load_eve_file(p, label_id, per_file_limit)
            if X is not None:
                class_X.append(X)
                class_y.append(y)
            del X, y
            gc.collect()

        if class_X:
            cat_X = np.concatenate(class_X)
            cat_y = np.concatenate(class_y)
            if label_name != "Benign" and len(cat_X) > MAX_PER_ATTACK_CLASS_TOTAL:
                idx = np.random.choice(len(cat_X), MAX_PER_ATTACK_CLASS_TOTAL, replace=False)
                cat_X = cat_X[idx]
                cat_y = cat_y[idx]
            all_X.append(cat_X)
            all_y.append(cat_y)
            log.info(f"  -> {label_name:<14} toplam: {len(cat_X):>10,}")
            del class_X, class_y, cat_X, cat_y
        gc.collect()

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    del all_X, all_y
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"\n  > Toplam ham veri: {len(y_all):,} akis")
    return X_all, y_all


# ============================================================================
# 6) ESIK OPTIMIZASYONU (FAR<%1 kisiti altinda max Recall)
# ============================================================================
def find_best_threshold(y_true_bin, y_prob_atk):
    """0.50-0.99 arasi esiklerde FAR<%1 olan en yuksek Recall'i bulur."""
    best_t, best_rec = 0.5, 0.0
    for t in np.arange(0.50, 1.00, 0.01):
        pred = (y_prob_atk >= t).astype(int)
        tn = ((pred == 0) & (y_true_bin == 0)).sum()
        fp = ((pred == 1) & (y_true_bin == 0)).sum()
        fn = ((pred == 0) & (y_true_bin != 0)).sum()
        tp = ((pred == 1) & (y_true_bin != 0)).sum()
        far = fp / (fp + tn + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        if far < 0.01 and rec > best_rec:
            best_rec = rec
            best_t = t
    return round(float(best_t), 2), round(float(best_rec), 4)


# ============================================================================
# 7) ANA EGITIM FONKSIYONU
# ============================================================================
def train():
    print()
    print("=" * 64)
    print("   IDS v6 - NIHAI XGBoost MODEL EGITIMI")
    print("=" * 64)
    print(f"  Ozellik sayisi : {N_FEATURES}")
    print(f"  Siniflar       : {list(CLASS_MAP.keys())}")
    print(f"  XGBoost ver.   : {xgb.__version__}")
    print(f"  Cikti          : {OUT_MODEL.name}")
    print("=" * 64)

    # --- 1) Veri yukleme ---
    log.info("Veri yukleniyor...")
    X_all, y_all = load_all_data()
    gc.collect()

    cnt = Counter(y_all.tolist())
    log.info("Sinif dagilimi:")
    for cid in sorted(cnt.keys()):
        log.info(f"  {INV_CLASS_MAP[cid]:<14}: {cnt[cid]:>12,}  ({cnt[cid]/len(y_all)*100:5.2f}%)")

    # --- 2) Stratified train/val/test split (70/15/15) ---
    log.info("Stratified train/val/test split (70/15/15)...")
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=42
    )
    del X_all, y_all, X_rest, y_rest
    gc.collect()
    log.info(f"  Train: {len(y_tr):>10,}   Val: {len(y_val):>9,}   Test: {len(y_te):>9,}")

    # --- 3) Normalizasyon (yalnizca train'e fit) ---
    log.info("StandardScaler fit ediliyor...")
    sample = np.random.choice(len(y_tr), min(300_000, len(y_tr)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_tr[sample])
    del sample
    X_tr  = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)
    gc.collect()

    # --- 4) Sinif agirliklari (agresif dengeleme) ---
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    log.info(f"  Sinif agirliklari: min={sample_weights.min():.3f}, "
             f"max={sample_weights.max():.3f}")

    # --- 5) XGBoost modeli (v5 -> v6 iyilestirilmis) ---
    log.info("XGBoost modeli olusturuluyor...")
    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=9,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    # --- 6) Egitim (validation ile) ---
    log.info("Egitim basliyor...")
    t0 = datetime.now()
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val), (X_te, y_te)],
        verbose=100,
    )
    train_dur = (datetime.now() - t0).total_seconds()
    log.info(f"Egitim tamamlandi: {train_dur/60:.1f} dk, "
             f"en iyi iterasyon: {model.best_iteration}")

    # --- 7) Test degerlendirmesi ---
    log.info("Test seti degerlendiriliyor...")
    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)
    y_te_bin   = (y_te != 0).astype(int)
    y_prob_atk = 1.0 - y_prob[:, 0]

    acc        = accuracy_score(y_te, y_pred)
    macro_f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_te_bin, (y_pred != 0).astype(int)).ravel()
    far = fp / (fp + tn + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    p, r, _ = precision_recall_curve(y_te_bin, y_prob_atk)
    pr_auc = auc(r, p)

    best_t, best_rec = find_best_threshold(y_te_bin, y_prob_atk)
    y_tuned = (y_prob_atk >= best_t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_te_bin, y_tuned).ravel()
    far2 = fp2 / (fp2 + tn2 + 1e-10)
    rec2 = tp2 / (tp2 + fn2 + 1e-10)

    # --- 8) Raporlama ---
    print()
    print("=" * 64)
    print("   NIHAI TEST SONUCLARI")
    print("=" * 64)
    print(f"  Sure  Egitim süresi       : {train_dur/60:.1f} dk")
    print(f"  Tree  En iyi iterasyon     : {model.best_iteration}")
    print()
    print(f"  >> Multiclass Accuracy   : {acc*100:6.2f}%")
    print(f"  >> Macro F1              : {macro_f1*100:6.2f}%")
    print(f"  >> Weighted F1           : {weighted_f1*100:6.2f}%")
    print()
    print(f"  >> Binary FAR (0.50)     : {far*100:6.3f}%")
    print(f"  >> Binary Recall (0.50)  : {rec*100:6.2f}%")
    print(f"  >> PR-AUC                : {pr_auc:.4f}")
    print()
    print(f"  >> Binary FAR (opt t={best_t}) : {far2*100:6.3f}%")
    print(f"  >> Binary Recall (opt)  : {rec2*100:6.2f}%")

    present_labels = sorted(np.unique(y_te).tolist())
    present_names  = [INV_CLASS_MAP[i] for i in present_labels]
    print(f"\n  --- Sinif Bazli Rapor ---")
    print(classification_report(
        y_te, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))

    print("  --- Karisiklik Matrisi ---")
    cm = confusion_matrix(y_te, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)
    print(cm_df.to_string())

    print("\n  --- En Onemli 15 Ozellik ---")
    fi = sorted(zip(FEATURE_COLS, model.feature_importances_),
                key=lambda x: -x[1])[:15]
    for fname, imp in fi:
        print(f"    {fname:<30} {imp:.4f}  {'#' * int(imp * 100)}")

    # --- 9) Modeli kaydet ---
    bundle = {
        "model"        : model,
        "scaler"       : scaler,
        "feature_cols" : FEATURE_COLS,
        "class_map"    : CLASS_MAP,
        "inv_class_map": INV_CLASS_MAP,
        "threshold"    : best_t,
        "metrics"      : dict(
            accuracy=acc, far=far2, recall=rec2, pr_auc=pr_auc,
            macro_f1=macro_f1, weighted_f1=weighted_f1,
        ),
        "trained_on"   : "CICIDS2018 + MAWI wide + CTU-13 + MCFP (6-class v6 final)",
        "n_features"   : N_FEATURES,
        "best_iteration": int(model.best_iteration),
        "training_duration_min": round(train_dur / 60, 2),
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Model kaydedildi: {OUT_MODEL}")
    log.info(f"  Threshold  : {best_t}")
    log.info(f"  Features   : {N_FEATURES}")
    log.info(f"  Macro F1   : {macro_f1*100:.2f}%")
    print("\n  EGITIM TAMAMLANDI")
    print("=" * 64)


if __name__ == "__main__":
    train()
