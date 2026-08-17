# -*- coding: utf-8 -*-
"""
egitim_v2.py - Eve.json Tabanlı IDS Eğitim Scripti (DÜZELTİLMİŞ)
===============================================================
Doğrudan Suricata eve.json dosyalarını okur.
CSV pipeline yok, COL_MAP yok, karmaşa yok.

Düzeltme: Stratified train/test split ile tüm sınıfların dağılımı korunur,
          num_class=7 açıkça belirtilir.

Sınıflar:
  0: Benign
  1: BruteForce
  2: DoS
  3: WebAttack
  4: Infiltration
  5: Bot
  6: DDoS

Çalıştırma:
    python3 egitim_v2.py
"""

import gc
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split          # ← EKLENDİ
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_curve, auc, classification_report
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── AYARLAR ───────────────────────────────────────────────────────────────────

BASE_DIR = Path("./pcap dosyaları ve veri setleri")
OUT_MODEL = Path("./ids_model_v2.pkl")

# Eve.json dosyaları ve label'ları
EVE_FILES = {
    "Benign"      : BASE_DIR / "Wednesday-14-02-2018/eve_Benign.json",
    "BruteForce"  : BASE_DIR / "Thursday-15-02-2018/eve_BruteForce.json",
    "DoS"         : [
        BASE_DIR / "Friday-16-02-2018/eve_DoS.json",
        BASE_DIR / "Tuesday-20-02-2018/eve_DoS.json",
        BASE_DIR / "Wednesday-21-02-2018/eve_DoS.json",
    ],
    "WebAttack"   : [
        BASE_DIR / "Thursday-22-02-2018/eve_WebAttack.json",
        BASE_DIR / "Friday-23-02-2018/eve_WebAttack.json",
    ],
    "Infiltration": BASE_DIR / "Wednesday-28-02-2018/eve_Infiltration.json",
    "Bot"         : BASE_DIR / "Thursday-01-03-2018/eve_Bot.json",
    "DDoS"        : BASE_DIR / "Friday-02-03-2018/eve_DDoS.json",
}

CLASS_MAP = {
    "Benign"      : 0,
    "BruteForce"  : 1,
    "DoS"         : 2,
    "WebAttack"   : 3,
    "Infiltration": 4,
    "Bot"         : 5,
    "DDoS"        : 6,
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
N_CLASSES = len(CLASS_MAP)   # 7

# Her sınıftan maksimum örnek (dengesizliği önlemek için)
MAX_PER_CLASS = 300_000

# ── FEATURE TANIMI ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Flow istatistikleri
    "duration_s",
    "pkts_toserver", "pkts_toclient",
    "bytes_toserver", "bytes_toclient",
    "total_pkts", "total_bytes",
    # Hız ve yoğunluk
    "pkt_rate", "byte_rate", "bytes_per_pkt",
    # Oran ve asimetri
    "upload_byte_ratio", "upload_pkt_ratio", "byte_asymmetry",
    "download_byte_ratio", "download_pkt_ratio",
    # Flow davranışı
    "flow_age",
    "is_unidirectional",       # pkts_toclient == 0
    "pkt_size_variance_proxy", # bytes/pkts toserver vs toclient farkı
    # Port bilgisi
    "dest_port", "src_port",
    "is_well_known_port",      # dst < 1024
    "is_registered_port",      # 1024 <= dst < 49152
    "is_high_port",            # dst >= 49152
    "is_same_port",            # src == dst
    # Network tipi
    "is_ipv6",
    "is_tcp", "is_udp", "is_icmp",
    # App protocol (one-hot)
    "app_http", "app_dns", "app_tls",
    "app_dcerpc", "app_smb", "app_rdp",
    "app_failed", "app_unknown",
    # Flow state
    "state_established", "state_closed", "state_new",
    # Flow reason
    "reason_timeout", "reason_rst", "reason_fin",
]

N_FEATURES = len(FEATURE_COLS)

# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_features(event: dict):
    """Eve.json flow event'inden feature vektörü çıkar."""
    if event.get("event_type") != "flow":
        return None

    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)

    # Duration
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
    sd  = max(dur, 1e-6)
    spk = max(tp, 1)
    sb  = max(tb, 1)

    # Protocol
    proto     = (event.get("proto") or "").upper()
    app_proto = (event.get("app_proto") or "unknown").lower()
    ip_v      = int(event.get("ip_v", 4) or 4)

    # Flow state/reason
    state  = (flow.get("state")  or "").lower()
    reason = (flow.get("reason") or "").lower()
    age    = float(flow.get("age", 0) or 0)

    features = [
        # Flow istatistikleri
        dur, pts, ptc, bts, btc, tp, tb,
        # Hız
        tp / sd, tb / sd, tb / spk,
        # Oranlar
        bts / sb, pts / spk, abs(bts - btc) / sb,
        btc / sb, ptc / spk,
        # Flow davranışı
        age,
        float(ptc == 0),                          # unidirectional
        abs((bts / max(pts, 1)) - (btc / max(ptc, 1))),  # pkt_size_variance_proxy
        # Port
        dp, sp,
        float(dp < 1024),
        float(1024 <= dp < 49152),
        float(dp >= 49152),
        float(sp == dp),
        # Network
        float(ip_v == 6),
        float(proto == "TCP"),
        float(proto == "UDP"),
        float(proto in ("ICMP", "ICMPv6")),
        # App proto
        float(app_proto == "http"),
        float(app_proto == "dns"),
        float(app_proto == "tls"),
        float(app_proto == "dcerpc"),
        float(app_proto == "smb"),
        float(app_proto == "rdp"),
        float(app_proto == "failed"),
        float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
        # State
        float(state == "established"),
        float(state == "closed"),
        float(state == "new"),
        # Reason
        float(reason == "timeout"),
        float(reason == "rst"),
        float(reason == "fin"),
    ]

    return np.array(features, dtype=np.float32)


# ── VERİ YÜKLEME ─────────────────────────────────────────────────────────────

def load_eve_file(path: Path, label_id: int, max_samples: int):
    """Tek eve.json dosyasından feature matrix yükle."""
    X, count = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if count >= max_samples:
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

        # Tek dosya veya liste
        if isinstance(paths, list):
            per_file = MAX_PER_CLASS // len(paths)
            file_list = paths
        else:
            per_file = MAX_PER_CLASS
            file_list = [paths]

        class_X, class_y = [], []
        for path in file_list:
            if not Path(path).exists():
                print(f"  [UYARI] Dosya bulunamadı: {path}")
                continue
            X, y = load_eve_file(Path(path), label_id, per_file)
            if X is not None:
                class_X.append(X)
                class_y.append(y)
                print(f"  {label_name:<15} ({path.name}): {len(X):,} örnek")

        if class_X:
            all_X.append(np.concatenate(class_X))
            all_y.append(np.concatenate(class_y))

        gc.collect()

    return np.concatenate(all_X), np.concatenate(all_y)


# ── THRESHOLD OPTİMİZASYONU ───────────────────────────────────────────────────

def find_best_threshold(y_true_bin, y_prob_atk):
    best_t, best_rec = 0.5, 0.0
    for t in np.arange(0.10, 0.95, 0.01):
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


# ── EĞİTİM ───────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("IDS EĞİTİM v2 - Eve.json Pipeline")
    print(f"Feature sayısı : {N_FEATURES}")
    print(f"Sınıflar       : {list(CLASS_MAP.keys())}")
    print(f"Max/sınıf      : {MAX_PER_CLASS:,}")
    print("=" * 60)

    print("\nVeri yükleniyor...\n")
    X_all, y_all = load_all_data()
    gc.collect()

    print(f"\nToplam: {len(y_all):,} örnek")
    u, c = np.unique(y_all, return_counts=True)
    for uu, cc in zip(u, c):
        pct = cc / len(y_all) * 100
        print(f"  {INV_CLASS_MAP[int(uu)]:<15}: {cc:,} ({pct:.1f}%)")

    # ═══════════════ DÜZELTME: STRATIFIED TRAIN/TEST SPLIT ═══════════════════
    # Artık zamansal dilimleme değil, tüm sınıfları orantılı dağıtan katmanlı bölme
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all,
        test_size=0.20,
        stratify=y_all,        # Sınıf dengesini koru
        random_state=42
    )
    del X_all, y_all
    gc.collect()

    print(f"\nEğitim : {len(y_tr):,}  |  Test: {len(y_te):,}")
    # ══════════════════════════════════════════════════════════════════════════

    # Scaler
    idx = np.random.choice(len(y_tr), min(200_000, len(y_tr)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_tr[idx])
    del idx
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    gc.collect()

    # Model (num_class=7 açıkça belirtilerek güvenceye alındı)
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        objective="multi:softprob",
        num_class=7,               # ← DÜZELTME: 7 sınıf için garanti
        eval_metric="mlogloss",
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    print("\nEğitim başlıyor...\n")
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

    # Değerlendirme
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

    # Kaydet
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
        "trained_on"   : "CICIDS2018-eve.json",
        "n_features"   : N_FEATURES,
    }
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n  Kaydedildi : {OUT_MODEL}")
    print(f"  Threshold  : {best_t}")
    print(f"  Features   : {N_FEATURES}")
    print("  Pi5 için: ids_model_v2.pkl + inference_v2.py")


if __name__ == "__main__":
    train()
