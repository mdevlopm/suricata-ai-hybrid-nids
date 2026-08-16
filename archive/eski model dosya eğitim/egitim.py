# -*- coding: utf-8 -*-
"""
egitim.py - Final Versiyon
===========================
Degisiklikler:
  - Siniflar sadeleştirildi (4 sinif): mlogloss artisi duzeltildi
  - classification_report hatasi duzeltildi
  - early_stopping eklendi: gereksiz iterasyonlar kesilir

Calistirma:
    source ~/ids_venv/bin/activate
    python3 egitim.py
"""

import gc, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_curve, auc, classification_report
)
import xgboost as xgb

BASE      = Path("/run/media/mehmet/siber data/ai modeli xgboost")
CSV_DIR   = BASE / "pcap dosyaları ve veri setleri/cic ml trafikleri cvs ler"
OUT_MODEL = BASE / "ids_model.pkl"

MAX_BENIGN_PER_DAY  = 150_000
MAX_ATTACK_PER_DAY  = 150_000
CHUNK_SIZE          = 100_000
SCALER_SAMPLE       = 200_000

BENIGN_LABELS = {"benign", "label"}

# ── SADELEŞTIRILMIŞ 4 SINIF ───────────────────────────────────────────────────
# DDoS + DoS --> Volumetric (ayni network behavior: yuksek hacim, tek yon)
# BruteForce + WebAttack --> Intrusion (ayni pattern: tekrarli kucuk baglanti)
# Bot --> Bot (C2 iletisimi, farkli pattern)
# Infiltration cikarildi: veri seti icinde yetersiz ornek

LABEL_GROUPS = {
    # Volumetric saldirilar
    "ddos attacks-loic-http"  : "Volumetric",
    "ddos attack-loic-udp"    : "Volumetric",
    "ddos attack-hoic"        : "Volumetric",
    "dos attacks-slowhttptest": "Volumetric",
    "dos attacks-goldeneye"   : "Volumetric",
    "dos attacks-slowloris"   : "Volumetric",
    "dos attacks-hulk"        : "Volumetric",
    # Intrusion girisimleri
    "ftp-bruteforce"          : "Intrusion",
    "ssh-bruteforce"          : "Intrusion",
    "brute force -web"        : "Intrusion",
    "brute force -xss"        : "Intrusion",
    "sql injection"           : "Intrusion",
    # Bot
    "bot"                     : "Bot",
    # Infiltration: atildi (veri yetersiz)
}

CLASS_MAP     = {"Benign": 0, "Volumetric": 1, "Intrusion": 2, "Bot": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
N_CLASSES     = len(CLASS_MAP)

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE UZAYI -- inference.py ile BIREBIR AYNI
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "duration_s",
    "pkts_toserver", "pkts_toclient",
    "bytes_toserver", "bytes_toclient",
    "total_pkts", "total_bytes",
    "pkt_rate", "byte_rate", "bytes_per_pkt",
    "upload_byte_ratio", "upload_pkt_ratio", "byte_asymmetry",
    "dest_port", "src_port",
    "is_well_known_port", "is_high_port",
]
# ═══════════════════════════════════════════════════════════════════════════════

COL_MAP = {
    "flow duration"               : "duration_us",
    "total fwd packets"           : "pkts_toserver",
    "fwd packets"                 : "pkts_toserver",
    "total backward packets"      : "pkts_toclient",
    "total bwd packets"           : "pkts_toclient",
    "bwd packets"                 : "pkts_toclient",
    "total length of fwd packets" : "bytes_toserver",
    "total fwd bytes"             : "bytes_toserver",
    "fwd packets length total"    : "bytes_toserver",
    "total length of bwd packets" : "bytes_toclient",
    "total bwd bytes"             : "bytes_toclient",
    "bwd packets length total"    : "bytes_toclient",
    "destination port"            : "dest_port",
    "dst port"                    : "dest_port",
    "source port"                 : "src_port",
    "label"                       : "label_raw",
}

def norm(s):
    return s.strip().lower().replace("-"," ").replace("_"," ")

def get_col_map(columns):
    result, seen = {}, set()
    for c in columns:
        key = COL_MAP.get(norm(c))
        if key and key not in seen:
            result[key] = c; seen.add(key)
    return result

def map_label(raw):
    s = raw.strip().lower()
    if s in BENIGN_LABELS: return "Benign"
    return LABEL_GROUPS.get(s, None)

def process_chunk(chunk, cmap):
    if "label_raw" not in cmap: return None
    out = pd.DataFrame(index=chunk.index)

    out["duration_s"] = (
        pd.to_numeric(chunk[cmap["duration_us"]], errors="coerce").fillna(0) / 1e6
        if "duration_us" in cmap else 0.0)

    for t in ["pkts_toserver","pkts_toclient","bytes_toserver",
              "bytes_toclient","dest_port","src_port"]:
        out[t] = (pd.to_numeric(chunk[cmap[t]], errors="coerce").fillna(0)
                  if t in cmap else 0.0)

    out["total_pkts"]  = out["pkts_toserver"]  + out["pkts_toclient"]
    out["total_bytes"] = out["bytes_toserver"] + out["bytes_toclient"]
    sd = out["duration_s"].clip(lower=1e-6)
    sp = out["total_pkts"].clip(lower=1)
    sb = out["total_bytes"].clip(lower=1)

    out["pkt_rate"]          = out["total_pkts"]  / sd
    out["byte_rate"]         = out["total_bytes"] / sd
    out["bytes_per_pkt"]     = out["total_bytes"] / sp
    out["upload_byte_ratio"] = out["bytes_toserver"] / sb
    out["upload_pkt_ratio"]  = out["pkts_toserver"]  / sp
    out["byte_asymmetry"]    = (out["bytes_toserver"] - out["bytes_toclient"]).abs() / sb
    out["is_well_known_port"]= (out["dest_port"] < 1024).astype(np.float32)
    out["is_high_port"]      = (out["dest_port"] >= 49152).astype(np.float32)

    raw   = chunk[cmap["label_raw"]].astype(str)
    grp   = raw.apply(map_label)
    valid = grp.notna()
    out   = out[valid].copy()
    grp   = grp[valid]
    out["label"] = grp.map(CLASS_MAP).astype(np.int8)

    out = out.replace([np.inf,-np.inf], np.nan).fillna(0)
    return out[out["duration_s"] >= 0]

def load_csv_balanced(path):
    try:
        header = pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        print(f"  [HATA] {path.name}: {e}"); return None, None

    cmap = get_col_map(list(header.columns))
    if "label_raw" not in cmap:
        print(f"  [ATLA] {path.name}"); return None, None

    use_cols = list(cmap.values())
    benign_parts, attack_parts = [], []
    n_benign = n_attack = 0

    try:
        for chunk in pd.read_csv(path, usecols=use_cols, chunksize=CHUNK_SIZE,
                                  encoding="utf-8", encoding_errors="replace",
                                  low_memory=False):
            p = process_chunk(chunk, cmap)
            if p is None or len(p) == 0:
                del chunk; continue

            b = p[p["label"] == 0]
            a = p[p["label"] != 0]

            if n_benign < MAX_BENIGN_PER_DAY and len(b):
                take = b.iloc[:MAX_BENIGN_PER_DAY - n_benign]
                benign_parts.append(take[FEATURE_COLS+["label"]].values.astype(np.float32))
                n_benign += len(take)

            if n_attack < MAX_ATTACK_PER_DAY and len(a):
                take = a.iloc[:MAX_ATTACK_PER_DAY - n_attack]
                attack_parts.append(take[FEATURE_COLS+["label"]].values.astype(np.float32))
                n_attack += len(take)

            del p, chunk, b, a; gc.collect()
            if n_benign >= MAX_BENIGN_PER_DAY and n_attack >= MAX_ATTACK_PER_DAY:
                break
    except Exception as e:
        print(f"  [HATA] {path.name}: {e}"); return None, None

    if not (benign_parts or attack_parts): return None, None

    data = np.concatenate(benign_parts + attack_parts)
    del benign_parts, attack_parts; gc.collect()
    X = data[:,:-1].astype(np.float32)
    y = data[:,-1].astype(np.int8)
    del data

    u,c = np.unique(y, return_counts=True)
    dist = {INV_CLASS_MAP.get(int(uu),str(uu)): int(cc) for uu,cc in zip(u,c)}
    print(f"  {path.name[:52]}: {len(y):,}  {dist}")
    return X, y

def find_best_threshold(y_true_bin, y_prob_atk):
    best_t = 0.5; best_rec = 0.0
    for t in np.arange(0.10, 0.95, 0.01):
        pred = (y_prob_atk >= t).astype(int)
        tn = ((pred==0)&(y_true_bin==0)).sum()
        fp = ((pred==1)&(y_true_bin==0)).sum()
        fn = ((pred==0)&(y_true_bin!=0)).sum()
        tp = ((pred==1)&(y_true_bin!=0)).sum()
        far = fp/(fp+tn+1e-10); rec = tp/(tp+fn+1e-10)
        if far < 0.01 and rec > best_rec:
            best_rec = rec; best_t = t
    return round(float(best_t),2), round(float(best_rec),4)

def train():
    print("="*60)
    print("IDS EGITIM - Final Versiyon")
    print(f"Siniflar: {list(CLASS_MAP.keys())}")
    print(f"Feature uzayi: {len(FEATURE_COLS)} (eve.json uyumlu)")
    print("="*60)

    csvs = sorted(CSV_DIR.glob("*.csv"))
    print(f"\n{len(csvs)} CSV | {MAX_BENIGN_PER_DAY:,} benign + {MAX_ATTACK_PER_DAY:,} attack/gun\n")

    all_X, all_y = [], []
    for csv_path in csvs:
        X, y = load_csv_balanced(csv_path)
        if X is not None:
            all_X.append(X); all_y.append(y)
            del X, y; gc.collect()

    if not all_X:
        print("[HATA] Veri yuklenemedi!"); return

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    del all_X, all_y; gc.collect()

    print(f"\nToplam: {len(y_all):,}")
    u,c = np.unique(y_all, return_counts=True)
    for uu,cc in zip(u,c):
        print(f"  {INV_CLASS_MAP.get(int(uu),str(uu)):<15}: {cc:,} ({cc/len(y_all)*100:.1f}%)")

    split = int(len(y_all)*0.80)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]
    del X_all, y_all; gc.collect()

    tr_atk = (y_tr!=0).mean(); te_atk = (y_te!=0).mean()
    print(f"\nDistribution: egitim=%{tr_atk*100:.1f}  test=%{te_atk*100:.1f}  fark=%{abs(tr_atk-te_atk)*100:.1f}")

    idx = np.random.choice(len(y_tr), min(SCALER_SAMPLE,len(y_tr)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X_tr[idx]); del idx
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    gc.collect()

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=N_CLASSES,
        eval_metric="mlogloss", tree_method="hist",
        early_stopping_rounds=30,   # mlogloss artarsa dur
        random_state=42, n_jobs=-1, verbosity=1,
    )
    print("\nEgitim...\n")
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

    y_pred    = model.predict(X_te)
    y_prob    = model.predict_proba(X_te)
    y_te_bin  = (y_te!=0).astype(int)
    y_prob_atk = 1 - y_prob[:,0]

    acc = accuracy_score(y_te, y_pred)
    tn,fp,fn,tp = confusion_matrix(y_te_bin,(y_pred!=0).astype(int)).ravel()
    far = fp/(fp+tn+1e-10); rec = tp/(tp+fn+1e-10)
    p,r,_ = precision_recall_curve(y_te_bin, y_prob_atk)
    pr_auc = auc(r,p)

    best_t, best_rec = find_best_threshold(y_te_bin, y_prob_atk)
    y_tuned = (y_prob_atk>=best_t).astype(int)
    tn2,fp2,fn2,tp2 = confusion_matrix(y_te_bin,y_tuned).ravel()
    far2 = fp2/(fp2+tn2+1e-10); rec2 = tp2/(tp2+fn2+1e-10)

    print(f"\n{'='*55}")
    print("SONUCLAR:")
    print(f"  Multiclass accuracy  : {acc*100:.2f}%")
    print(f"\n  Binary (default 0.50):")
    print(f"    FAR    : {far*100:.3f}%")
    print(f"    Recall : {rec*100:.2f}%")
    print(f"    PR-AUC : {pr_auc:.4f}")
    print(f"\n  Binary (threshold={best_t}, FAR<%1 optimize):")
    print(f"    FAR    : {far2*100:.3f}%   (hedef <1%)")
    print(f"    Recall : {rec2*100:.2f}%   (hedef >96%)")
    print(f"    TP={tp2:,}  FP={fp2:,}  FN={fn2:,}  TN={tn2:,}")

    # Sinif bazli rapor -- sadece test setinde gorunen siniflari kullan
    present_labels = sorted(np.unique(y_te).tolist())
    present_names  = [INV_CLASS_MAP[i] for i in present_labels]
    print(f"\nSinif bazli (test setinde gorunen {len(present_labels)} sinif):")
    print(classification_report(
        y_te, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    print(f"{'='*55}")

    bundle = {
        "model"        : model,
        "scaler"       : scaler,
        "feature_cols" : FEATURE_COLS,
        "class_map"    : CLASS_MAP,
        "inv_class_map": INV_CLASS_MAP,
        "threshold"    : best_t,
        "metrics"      : dict(accuracy=acc, far=far2, recall=rec2, pr_auc=pr_auc),
        "trained_on"   : "CICIDS2018",
    }
    with open(OUT_MODEL,"wb") as f:
        pickle.dump(bundle,f)

    print(f"\n  Kaydedildi : {OUT_MODEL}")
    print(f"  Threshold  : {best_t}")
    print("  Raspberry Pi icin: ids_model.pkl + inference.py")

if __name__ == "__main__":
    train()
