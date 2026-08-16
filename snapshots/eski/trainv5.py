# -*- coding: utf-8 -*-
"""
Eğitim v5 – Omurga (wide) verileri Benign'e eklendi.
Bellek Optimizasyonlu + Feature Düzeltme + Overfitting Önleme
"""

import gc, json, pickle, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_curve, auc, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
warnings.filterwarnings("ignore")

BASE_DIR = Path("/run/media/mehmet/siber data1/ai modeli xgboost/pcap dosyaları ve veri setleri")
OUT_MODEL = Path("/run/media/mehmet/siber data1/ai modeli xgboost/ids_model_v5_optimized.pkl")

EVE_FILES = {
    "Benign"      : [   # Liste hâline getirildi
        BASE_DIR / "Wednesday-14-02-2018/eve_Benign.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407021400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202407071400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408111400.json",
        BASE_DIR / "Omurga verisi wide/suricata_202408121400.json",
    ],
    "DoS"         : [
        BASE_DIR / "Thursday-15-02-2018/eve_DoS.json",
        BASE_DIR / "Friday-16-02-2018/eve_DoS.json",
    ],
    "DDoS"        : [
        BASE_DIR / "Tuesday-20-02-2018/eve_DDoS.json",
        BASE_DIR / "Wednesday-21-02-2018/eve_DDoS.json",
    ],
    "WebAttack"   : [
        BASE_DIR / "Thursday-22-02-2018/eve_WebAttack.json",
        BASE_DIR / "Friday-23-02-2018/eve_WebAttack.json",
    ],
    "Infiltration": [
        BASE_DIR / "Wednesday-28-02-2018/eve_Infiltration.json",
        BASE_DIR / "Thursday-01-03-2018/eve_Infiltration.json",
    ],
    "Bot"         : BASE_DIR / "Friday-02-03-2018/eve_Bot.json",
}

CLASS_MAP = {
    "Benign"      : 0,
    "DoS"         : 1,
    "DDoS"        : 2,
    "WebAttack"   : 3,
    "Infiltration": 4,
    "Bot"         : 5,
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
N_CLASSES = len(CLASS_MAP)

# --- SINIR AYARLARI ---
# Omurga verisiyle birlikte Benign sayısı çok artacak.
# RAM yetiyorsa None yapabilirsiniz, yoksa aşağıdaki gibi bir üst sınır koyun.
MAX_BENIGN = 20_000_000          # None yaparak sınırsız okuyabilirsiniz
MAX_PER_ATTACK_CLASS = 50_000_000  # Saldırı sınıfları için üst sınır (aynı kaldı)

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
    "is_well_known_port",
    "is_registered_port",
    "is_high_port",
    "is_same_port",
    "is_ipv6",
    "is_tcp", "is_udp", "is_icmp",
    "app_http", "app_dns", "app_tls",
    "app_dcerpc", "app_smb", "app_rdp",
    "app_failed", "app_unknown",
    "state_established", "state_closed", "state_new",
    "reason_timeout", "reason_rst", "reason_fin",
]
N_FEATURES = len(FEATURE_COLS)

def extract_features(event: dict):
    if event.get("event_type") != "flow":
        return None
    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)
    try:
        t0 = datetime.fromisoformat(
            (flow.get("start") or event.get("timestamp", "")).replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(
            (flow.get("end") or "").replace("Z", "+00:00"))
        dur = max((t1 - t0).total_seconds(), 0.0)
    except:
        dur = 0.0
    dp  = int(event.get("dest_port", 0) or 0)
    sp  = int(event.get("src_port",  0) or 0)
    tp  = pts + ptc
    tb  = bts + btc

    # -- DÜZELTME: Çok küçük sürelerde hızları sınırlandır --
    MIN_DUR = 0.1
    sd_clipped = max(dur, MIN_DUR)
    # -----------------------------------------------------

    spk = max(tp, 1)
    sb  = max(tb, 1)
    proto     = (event.get("proto") or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)
    state     = (flow.get("state")  or "").lower()
    reason    = (flow.get("reason") or "").lower()
    age       = float(flow.get("age", 0) or 0)

    features = [
        dur, pts, ptc, bts, btc, tp, tb,
        tp / sd_clipped,          # pkt_rate
        tb / sd_clipped,          # byte_rate
        tb / spk,                 # bytes_per_pkt
        bts / sb,                 # upload_byte_ratio
        pts / spk,                # upload_pkt_ratio
        abs(bts - btc) / sb,      # byte_asymmetry
        btc / sb,                 # download_byte_ratio
        ptc / spk,                # download_pkt_ratio
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
        float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state == "established"),
        float(state == "closed"),
        float(state == "new"),
        float(reason == "timeout"),
        float(reason == "rst"),
        float(reason == "fin"),
    ]
    return np.array(features, dtype=np.float32)

def load_eve_file(path: Path, label_id: int, max_samples: int):
    X, count = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_samples is not None and count >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                feat = extract_features(event)
                if feat is not None:
                    X.append(feat)
                    count += 1
            except:
                continue
    if not X:
        return None, None
    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.full(len(X_arr), label_id, dtype=np.int8)
    return X_arr, y_arr

def load_all_data():
    all_X, all_y = [], []
    for label_name, paths in EVE_FILES.items():
        label_id = CLASS_MAP[label_name]
        if label_name == "Benign":
            max_for_class = MAX_BENIGN
        else:
            max_for_class = MAX_PER_ATTACK_CLASS

        # Dosya listesini ayarla
        if isinstance(paths, list):
            file_list = paths
        else:
            file_list = [paths]

        # Dosya başına örnek sayısı
        if max_for_class is None:
            per_file = None          # sınırsız
        else:
            per_file = max_for_class // len(file_list)

        class_X, class_y = [], []
        for path in file_list:
            if not Path(path).exists():
                print(f"  [UYARI] Dosya bulunamadı: {path}")
                continue
            X, y = load_eve_file(Path(path), label_id, per_file)
            if X is not None:
                class_X.append(X)
                class_y.append(y)
                print(f"  {label_name:<15} ({Path(path).name}): {len(X):,} örnek")
            del X, y
            gc.collect()
        if class_X:
            all_X.append(np.concatenate(class_X))
            all_y.append(np.concatenate(class_y))
            del class_X, class_y
        gc.collect()

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    del all_X, all_y
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    gc.collect()
    return X_all, y_all

def find_best_threshold(y_true_bin, y_prob_atk):
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

def train():
    print("=" * 60)
    print("IDS EĞİTİM v5 – Omurga verisi Benign'e eklendi")
    print(f"Feature sayısı : {N_FEATURES}")
    print(f"Sınıflar       : {list(CLASS_MAP.keys())}")
    print(f"Benign üst sınır: {MAX_BENIGN if MAX_BENIGN is not None else 'SINIRSIZ'}")
    print(f"Saldırı üst sınır: {MAX_PER_ATTACK_CLASS:,}")
    print("=" * 60)

    print("\nVeri yükleniyor...\n")
    X_all, y_all = load_all_data()
    gc.collect()

    print(f"\nToplam: {len(y_all):,} örnek")
    u, c = np.unique(y_all, return_counts=True)
    for uu, cc in zip(u, c):
        pct = cc / len(y_all) * 100
        print(f"  {INV_CLASS_MAP[int(uu)]:<15}: {cc:,} ({pct:.1f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
    )
    del X_all, y_all
    gc.collect()
    print(f"\nEğitim : {len(y_tr):,}  |  Test: {len(y_te):,}")

    idx = np.random.choice(len(y_tr), min(200_000, len(y_tr)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_tr[idx])
    del idx
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    gc.collect()

    sample_weights = compute_sample_weight('balanced', y_tr)

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.3,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method='hist',
        max_bin=128,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print("\nEğitim başlıyor...\n")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        sample_weight=sample_weights,
        verbose=100
    )

    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)
    y_te_bin   = (y_te != 0).astype(int)
    y_prob_atk = 1 - y_prob[:, 0]

    acc = accuracy_score(y_te, y_pred)
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

    print(f"\n{'=' * 55}")
    print("SONUÇLAR:")
    print(f"  Multiclass accuracy  : {acc * 100:.2f}%")
    print(f"\n  Binary (default 0.50):")
    print(f"    FAR    : {far * 100:.3f}%")
    print(f"    Recall : {rec * 100:.2f}%")
    print(f"    PR-AUC : {pr_auc:.4f}")
    print(f"\n  Binary (threshold={best_t}, FAR<%1 optimize):")
    print(f"    FAR    : {far2 * 100:.3f}%")
    print(f"    Recall : {rec2 * 100:.2f}%")
    print(f"    TP={tp2:,}  FP={fp2:,}  FN={fn2:,}  TN={tn2:,}")

    present_labels = sorted(np.unique(y_te).tolist())
    present_names  = [INV_CLASS_MAP[i] for i in present_labels]
    print(f"\nSınıf bazlı rapor:")
    print(classification_report(
        y_te, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    print(f"{'=' * 55}")

    bundle = {
        "model"        : model,
        "scaler"       : scaler,
        "feature_cols" : FEATURE_COLS,
        "class_map"    : CLASS_MAP,
        "inv_class_map": INV_CLASS_MAP,
        "threshold"    : best_t,
        "metrics"      : dict(
            accuracy=acc, far=far2, recall=rec2, pr_auc=pr_auc
        ),
        "trained_on"   : "CICIDS2018 + MAWI wide backbone (6-class optimized v5)",
        "n_features"   : N_FEATURES,
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Kaydedildi : {OUT_MODEL}")
    print(f"  Threshold  : {best_t}")
    print(f"  Features   : {N_FEATURES}")

if __name__ == "__main__":
    train()
