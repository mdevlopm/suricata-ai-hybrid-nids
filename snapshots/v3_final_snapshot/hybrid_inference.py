#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_inference.py - Hibrit IDS Cikarim Motoru (XGBoost + LSTM) — v2
=======================================================================
Guncelleme (v2):
  - WINDOW_SIZE 20 → 40  (lstm_train.py ile eslestirildi)
  - extract_features() → extract_features_v7() (70 ozellik, FlowEnrichment)
  - DoS(1)/DDoS(2): XGBoost tespiti yeterli → LSTM atlanir, "DoS/DDoS" alarmi
  - LSTM_INV_CLASS_MAP: {0:Volumetric, 1:WebAttack, 2:Bot}
    Volumetric = DoS + DDoS + Infiltration (literatur destekli)
  - Hatali +1 label shift kaldirildi

Mimari:
  Suricata eve.json
       |
       v
  extract_features_v7()  →  70 ozellik (42 base + 28 zengin)
       |
       v
  XGBoost (v6/v7) ──Benign(0)──────────────────────→ yok say
                 ├──DoS(1) / DDoS(2)──────────────→ "DoS/DDoS" alarmi
                 │                                   LSTM ATLA
                 └──WebAttack(3) / Infiltration(4) / Bot(5)
                           |
                           v
                     IPBuffer (src_ip basina son 40 akis)
                           |  (40 akis dolunca)
                           v
                     LSTM → Volumetric / WebAttack / Bot

Kullanim:
  Gercek zamanli:
    python3 hybrid_inference.py --eve /var/log/suricata/eve.json

  Batch:
    python3 hybrid_inference.py --eve test.json --batch --output ./alerts.json

  Model yolu:
    python3 hybrid_inference.py \\
        --xgb_model ./ids_model_v6_final.pkl \\
        --lstm_model ./lstm_best.keras \\
        --lstm_scaler ./lstm_scaler.pkl
"""

import argparse, json, pickle, time, sys, os
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ip_buffer import IPBuffer
from features import compute_ip_window_features

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ============================================================================
# VARSAYILAN YOLLAR
# ============================================================================
DEFAULT_EVE         = Path("/var/log/suricata/eve.json")
DEFAULT_OUT         = Path("/var/log/suricata/ai_alerts_hybrid.json")
DEFAULT_XGB_MODEL   = Path("./ids_model_v7_final.pkl")
DEFAULT_LSTM_MODEL  = Path(__file__).parent / "lstm_best.keras"
DEFAULT_LSTM_SCALER = Path(__file__).parent / "lstm_scaler.pkl"
DEFAULT_LSTM_META   = Path(__file__).parent / "lstm_metadata.pkl"

POLL_S               = 0.5
CLEANUP_INTERVAL     = 1000
BUFFER_TIMEOUT_S     = 300
WINDOW_SIZE          = 40              # lstm_train.py ile eslestirildi (20 degil)
XGB_THRESHOLD_FALLBACK = 0.59
LSTM_CONF_THRESHOLD  = 0.5             # Kararsiz LSTM tahminleri icin esik

# DoS=1, DDoS=2 → XGBoost sonucu yeterli, LSTM'e gitmez
XGB_DOS_DDOS_CLASSES = {1, 2}

# LSTM 3 sinif: argmax indeksinden etiket (0-indexed, +1 yok)
# Volumetric = DoS + DDoS + Infiltration
LSTM_INV_CLASS_MAP = {0: "Volumetric", 1: "WebAttack", 2: "Bot"}


# ============================================================================
# FLOW ZENGINLESTIRME (http/tls/dns eventlerini flow_id ile onbellege al)
# ============================================================================
class FlowEnrichment:
    """Single-pass: http/tls/dns eventleri flow_id bazinda onbellege alir.
    Zaman bazli otomatik temizlik ile bellek sizintilari onlenir."""

    def __init__(self, max_entries: int = 200_000, timeout_s: int = 300):
        self._store = {}
        self._max   = max_entries
        self.timeout_s = timeout_s
        self.ingest_counter = 0

    def ingest(self, event: dict):
        etype = event.get("event_type")
        if etype not in ("http", "tls", "dns"):
            return
        fid = event.get("flow_id")
        if fid is None:
            return

        # Periyodik temizlik
        self.ingest_counter += 1
        if self.ingest_counter % 5000 == 0:
            self.cleanup()

        if len(self._store) >= self._max:
            self.cleanup()
            if len(self._store) >= self._max:
                return

        entry = self._store.setdefault(fid, {"_ts": datetime.now()})
        if etype not in entry:
            entry[etype] = event

    def get(self, flow_id):
        """flow_id'ye ait enrichment dict'ini alir ve onbellekten siler."""
        entry = self._store.pop(flow_id, None) if flow_id else None
        if entry:
            entry.pop("_ts", None)
            return entry
        return None

    def cleanup(self):
        """Eski yetim kalmis http/tls/dns kayitlarini temizler."""
        now = datetime.now()
        expired = [
            fid for fid, entry in self._store.items()
            if (now - entry["_ts"]).total_seconds() > self.timeout_s
        ]
        for fid in expired:
            del self._store[fid]

    def clear(self):
        self._store.clear()


# ============================================================================
# OZELLIK CIKARIMI — v7  (42 base + 28 zenginlestirilmis = 70 ozellik)
# ============================================================================
def _extract_tcp_flags(event: dict) -> dict:
    tcp = event.get("tcp") if event else None
    if not tcp:
        return {"syn": 0, "ack": 0, "fin": 0, "rst": 0, "psh": 0}
    try:
        flags_int = int(tcp.get("tcp_flags_ts", "00"), 16)
    except (ValueError, TypeError):
        flags_int = 0
    return {
        "syn": float(bool(flags_int & 0x02)),
        "ack": float(bool(flags_int & 0x10)),
        "fin": float(bool(flags_int & 0x01)),
        "rst": float(bool(flags_int & 0x04)),
        "psh": float(bool(flags_int & 0x08)),
    }


def _extract_http_features(enrichment: dict) -> dict:
    http_ev = enrichment.get("http") if enrichment else None
    if not http_ev:
        return {"mo": [0,0,0,0], "so": [0,0,0,0],
                "cth": 0, "cto": 0, "ctj": 0, "ct_": 0}
    h  = http_ev.get("http", {})
    m  = (h.get("http_method") or "").upper()
    mv = [0,0,0,0]
    if m == "GET":    mv[0] = 1
    elif m == "POST": mv[1] = 1
    elif m == "HEAD": mv[2] = 1
    elif m:           mv[3] = 1
    s  = int(h.get("status", 0) or 0)
    sv = [0,0,0,0]
    if   200 <= s < 300: sv[0] = 1
    elif 300 <= s < 400: sv[1] = 1
    elif 400 <= s < 500: sv[2] = 1
    elif s >= 500:       sv[3] = 1
    ct = (h.get("http_content_type") or "").lower()
    return {
        "mo": mv, "so": sv,
        "cth": float("html" in ct),
        "cto": float("octet-stream" in ct or "binary" in ct),
        "ctj": float("json" in ct),
        "ct_": float(bool(ct) and
                     not ("html" in ct or "octet-stream" in ct
                          or "binary" in ct or "json" in ct)),
    }


def _extract_tls_features(enrichment: dict) -> dict:
    tls_ev = enrichment.get("tls") if enrichment else None
    if not tls_ev:
        return {"e": 0, "sni": 0}
    sni = (tls_ev.get("tls", {}).get("sni") or "")
    return {"e": 1, "sni": float(bool(sni))}


def _extract_dns_features(enrichment: dict) -> dict:
    dns_ev = enrichment.get("dns") if enrichment else None
    if not dns_ev:
        return {"e": 0, "qa": 0, "qaa": 0, "qm": 0, "qo": 0,
                "rn": 0, "rnx": 0, "rr": 0, "ro": 0}
    d  = dns_ev.get("dns", {})
    qt = set((q.get("rrtype") or "").upper()
             for q in (d.get("queries") or []))
    rc = (d.get("rcode") or "").upper()
    return {
        "e":  1,
        "qa": float("A" in qt), "qaa": float("AAAA" in qt),
        "qm": float("MX" in qt), "qo": float(bool(qt - {"A","AAAA","MX"})),
        "rn": float(rc == "NOERROR"), "rnx": float(rc == "NXDOMAIN"),
        "rr": float(rc == "REFUSED"),
        "ro": float(bool(rc) and rc not in ("NOERROR","NXDOMAIN","REFUSED")),
    }


def extract_features_v7(event: dict, enrichment: dict):
    """Suricata flow event'inden 70 ozellik cikarir.
    Donus: (np.ndarray shape (1,70), meta dict)  veya  (None, None)"""
    if event.get("event_type") != "flow":
        return None, None

    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)

    try:
        from dateutil.parser import isoparse
        t0  = isoparse(
            (flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1  = isoparse(
            (flow.get("end") or "").replace("Z","+00:00"))
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

    # ── 42 temel ozellik ──────────────────────────────────────────────────
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
        float(app_proto not in
              ("http","dns","tls","dcerpc","smb","rdp","failed")),
        float(state == "established"), float(state == "closed"),
        float(state == "new"),
        float(reason == "timeout"), float(reason == "rst"),
        float(reason == "fin"),
    ], dtype=np.float32)

    # ── 28 zenginlestirilmis ozellik ─────────────────────────────────────
    enriched = np.zeros(28, dtype=np.float32)
    tcpf = _extract_tcp_flags(event)
    idx  = 0
    enriched[idx:idx+5] = [tcpf["syn"], tcpf["ack"],
                            tcpf["fin"], tcpf["rst"], tcpf["psh"]]
    idx += 5

    if enrichment:
        hf = _extract_http_features(enrichment)
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
        enriched[idx:idx+4] = [dnsf["qa"], dnsf["qaa"],
                                dnsf["qm"], dnsf["qo"]];  idx += 4
        enriched[idx:idx+4] = [dnsf["rn"], dnsf["rnx"],
                                dnsf["rr"], dnsf["ro"]];  idx += 4

    features = np.concatenate([base, enriched]).reshape(1, -1)

    meta = {
        "timestamp"  : event.get("timestamp", ""),
        "src_ip"     : event.get("src_ip",  ""),
        "dest_ip"    : event.get("dest_ip", ""),
        "src_port"   : sp,
        "dest_port"  : dp,
        "proto"      : proto,
        "app_proto"  : app_proto,
        "flow_id"    : event.get("flow_id", ""),
        "duration_s" : round(dur, 3),
        "total_bytes": int(tb),
        "byte_rate"  : round(tb / sd, 1),
    }
    return features, meta


# ============================================================================
# IP-WINDOW META CIKARIMI (Bot/WebAttack behavioral features icin)
# ============================================================================
def _extract_dns_queries(enrichment):
    if not enrichment: return []
    dns_ev = enrichment.get("dns")
    if not dns_ev: return []
    dns = dns_ev.get("dns", {})
    return [q.get("rrname", "") for q in (dns.get("queries") or []) if q.get("rrname")]

def _extract_http_uri(enrichment):
    if not enrichment: return None
    http_ev = enrichment.get("http")
    if not http_ev: return None
    return http_ev.get("http", {}).get("url", "")

def _extract_tls_sni(enrichment):
    if not enrichment: return None
    tls_ev = enrichment.get("tls")
    if not tls_ev: return None
    return tls_ev.get("tls", {}).get("sni", "")


# ============================================================================
# IP BUFFER (Imported from ip_buffer.py)
# ============================================================================



# ============================================================================
# YARDIMCI: ALARM SOZLUGU OLUSTUR
# ============================================================================
def _build_alert(meta, label, confidence, stage, xgb_class, xgb_label):
    return {
        "timestamp"  : meta["timestamp"],
        "event_type" : "ai_alert_hybrid",
        "src_ip"     : meta["src_ip"],
        "dest_ip"    : meta["dest_ip"],
        "src_port"   : meta["src_port"],
        "dest_port"  : meta["dest_port"],
        "proto"      : meta["proto"],
        "app_proto"  : meta["app_proto"],
        "flow_id"    : meta["flow_id"],
        "ai": {
            "label"      : label,
            "confidence" : confidence,
            "stage"      : stage,
            "xgb_class"  : xgb_class,
            "xgb_label"  : xgb_label,
            "duration_s" : meta["duration_s"],
            "total_bytes": meta["total_bytes"],
            "byte_rate"  : meta["byte_rate"],
        }
    }


# ============================================================================
# MODEL YUKLEME
# ============================================================================
def load_xgboost(path: Path):
    print(f"  XGBoost model: {path}")
    with open(path, "rb") as f:
        b = pickle.load(f)
    model    = b["model"]
    scaler   = b["scaler"]
    thresh   = b.get("threshold", XGB_THRESHOLD_FALLBACK)
    inv_cmap = b.get("inv_class_map", {})
    cmap     = b.get("class_map", {})
    n_classes = len(cmap)
    metrics  = b.get("metrics", {})
    print(f"    Threshold : {thresh}")
    print(f"    Sinif     : {n_classes}")
    print(f"    FAR       : {metrics.get('far',0)*100:.3f}%")
    print(f"    Recall    : {metrics.get('recall',0)*100:.2f}%")
    print(f"    Macro F1  : {metrics.get('macro_f1',0)*100:.2f}%")
    is_multiclass = n_classes > 2
    if not is_multiclass:
        print(f"    !! Binary model -> xgb_only'da 'Generic Attack' kullanilacak")
    # CUDA / CPU cihaz uyumu
    try:
        model.set_params(device="cpu")
        model.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass
    return model, scaler, thresh, inv_cmap, cmap, is_multiclass


def load_lstm(model_path: Path, scaler_path: Path, meta_path: Path):
    if not model_path.exists():
        print("  !! LSTM model bulunamadi -> sadece XGBoost modunda calis")
        return None, None, None
    print(f"  LSTM model  : {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"    Window    : {meta.get('window_size','?')}")
        print(f"    Features  : {meta.get('n_features','?')}")
        print(f"    Classes   : {meta.get('lstm_classes','?')}")
    print(f"    Params    : {sum(np.prod(w.shape) for w in model.weights):,}")
    return model, scaler, meta


# ============================================================================
# HIBRIT CIKARIM
# ============================================================================
def hybrid_predict(event, enrichment,
                   xgb_model, xgb_scaler, xgb_threshold,
                   inv_class_map, class_map, is_multiclass,
                   lstm_model, lstm_scaler, lstm_meta,
                   ip_buffer, coral_adapter=None):
    """Tek bir flow event'i isler: Feature Extraction → (CORAL alignment) → XGBoost → IPBuffer → LSTM → Alert"""

    features, meta = extract_features_v7(event, enrichment)
    if features is None:
        return None

    # ── 0) CORAL Domain Adaptation (Target -> Source Alignment) ───────────────
    if coral_adapter is not None:
        try:
            features = coral_adapter.transform_target(features)
        except Exception:
            pass

    # ── 1) XGBoost ──────────────────────────────────────────────────────────
    xgb_prob = xgb_model.predict_proba(xgb_scaler.transform(features))[0]
    prob_atk = 1.0 - xgb_prob[0]
    xgb_pred = int(xgb_prob.argmax())

    # Esik alti veya Benign → yok say
    if prob_atk < xgb_threshold:
        return None
    if is_multiclass and xgb_pred == 0:
        return None

    src_ip    = meta["src_ip"]
    xgb_label = (inv_class_map.get(xgb_pred, "Unknown")
                 if is_multiclass else "Generic Attack")

    # ── 2) DoS / DDoS → LSTM ATLA ───────────────────────────────────────────
    if is_multiclass and xgb_pred in XGB_DOS_DDOS_CLASSES:
        return _build_alert(
            meta,
            label      = "DoS/DDoS",
            confidence = round(float(prob_atk), 4),
            stage      = "xgb_only",
            xgb_class  = xgb_pred,
            xgb_label  = xgb_label,
        )

    # ── 3) IP-window meta cikarimi ──────────────────────────────────────────
    flow_data = event.get("flow", {})
    bts = float(flow_data.get("bytes_toserver", 0) or 0)
    btc = float(flow_data.get("bytes_toclient", 0) or 0)
    ip_meta = {
        "ts": datetime.now(),
        "dest_ip": event.get("dest_ip", ""),
        "dest_port": int(event.get("dest_port", 0) or 0),
        "total_bytes": int(bts + btc),
        "dns_queries": _extract_dns_queries(enrichment),
        "http_uri": _extract_http_uri(enrichment),
        "tls_sni": _extract_tls_sni(enrichment),
    }

    # ── 4) Diger saldirilar → IPBuffer'a ekle ───────────────────────────────
    ip_buffer.add(src_ip, features[0], ip_meta, timestamp=datetime.now())

    # ── 5) LSTM penceresi dolu mu? ──────────────────────────────────────────
    window, _ = ip_buffer.get_window(src_ip)

    if window is not None and lstm_model is not None:
        # 78 ozellik × 40 akis penceresi → normalize → LSTM
        window_norm = lstm_scaler.transform(
            window.reshape(-1, window.shape[-1]))
        window_norm = window_norm.reshape(1, WINDOW_SIZE, -1).astype(np.float16)

        lstm_prob  = lstm_model.predict(window_norm, verbose=0)[0]
        lstm_idx   = int(np.argmax(lstm_prob))          # 0, 1 veya 2
        lstm_conf  = float(lstm_prob[lstm_idx])
        if lstm_conf < LSTM_CONF_THRESHOLD:
            label  = "Generic Attack"
        else:
            label  = LSTM_INV_CLASS_MAP.get(lstm_idx, "Unknown")
        stage      = "xgb+lstm"
        confidence = round(lstm_conf, 4)
    else:
        # Pencere henuz dolmadi → sadece XGBoost sonucu
        label      = xgb_label
        stage      = "xgb_only"
        confidence = round(float(prob_atk), 4)

    return _build_alert(meta, label, confidence, stage, xgb_pred, xgb_label)


# ============================================================================
# BATCH MOD
# ============================================================================
def run_batch(eve_path, xgb_model, xgb_scaler, xgb_threshold,
              inv_class_map, class_map, is_multiclass,
              lstm_model, lstm_scaler, lstm_meta, output_path,
              coral_adapter=None):
    print(f"\n  BATCH MOD (HIBRIT v2)")
    print(f"  Eve.json : {eve_path}")
    print(f"  Cikti    : {output_path}")
    if coral_adapter is not None:
        print(f"  CORAL    : Aktif (Domain Adaptation Alignment)")
    print()

    ip_buffer  = IPBuffer()
    flow_cache = FlowEnrichment()      # http/tls/dns onbellegi
    stats      = {"total": 0, "attack": 0, "benign": 0, "skip": 0}
    attack_types = defaultdict(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as out:
        with open(eve_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    stats["skip"] += 1
                    continue

                etype = event.get("event_type")

                # http/tls/dns → onbellege al, sınıflandırma yok
                if etype in ("http", "tls", "dns"):
                    flow_cache.ingest(event)
                    continue

                # Flow olmayan diger eventler → atla
                if etype != "flow":
                    stats["skip"] += 1
                    continue

                stats["total"] += 1
                enrichment = flow_cache.get(event.get("flow_id"))

                alert = hybrid_predict(
                    event, enrichment,
                    xgb_model, xgb_scaler, xgb_threshold,
                    inv_class_map, class_map, is_multiclass,
                    lstm_model, lstm_scaler, lstm_meta,
                    ip_buffer, coral_adapter=coral_adapter,
                )

                if alert:
                    stats["attack"] += 1
                    lbl = alert["ai"]["label"]
                    attack_types[lbl] += 1
                    out.write(json.dumps(alert) + "\n")

                    if stats["total"] <= 10:
                        print(f"  [ALERT] {alert['timestamp']} | "
                              f"{lbl:<14} | "
                              f"conf={alert['ai']['confidence']:.3f} | "
                              f"stage={alert['ai']['stage']}")
                else:
                    stats["benign"] += 1

                if stats["total"] % 100_000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  ... {stats['total']:,} flow islendi  "
                          f"({stats['total']/max(elapsed,0.001):,.0f} flow/s)")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  BATCH SONUCU (HIBRIT v2)")
    print(f"{'='*60}")
    print(f"  Toplam flow : {stats['total']:>12,}")
    print(f"  Benign      : {stats['benign']:>12,}")
    print(f"  Alert       : {stats['attack']:>12,}  "
          f"({stats['attack']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Gecersiz    : {stats['skip']:>12,}")
    print(f"  Sure        : {elapsed/60:.1f} dk  "
          f"({stats['total']/max(elapsed,0.001):,.0f} flow/s)")
    if attack_types:
        print(f"\n  Saldiri Turleri:")
        for t, c in sorted(attack_types.items(), key=lambda x: -x[1]):
            print(f"    {t:<16}: {c:,}")
    print(f"\n  Kaydedildi  : {output_path}")
    print(f"{'='*60}")


# ============================================================================
# GERCEK ZAMANLI MOD
# ============================================================================
class EveJsonTailer:
    """Suricata eve.json'u log rotasyonunu da takip ederek okur (inode)."""
    def __init__(self, path):
        self.path   = path
        self._f     = None
        self._inode = None
        self._open()

    def _open(self):
        if self._f:
            self._f.close()
        if self.path.exists():
            self._f     = open(self.path, encoding="utf-8", errors="replace")
            self._f.seek(0, 2)   # dosya sonuna git
            self._inode = self.path.stat().st_ino
        else:
            self._f = self._inode = None

    def readlines(self):
        if not self.path.exists():
            if self._f:
                self._f.close()
                self._f = None
            return []
        if self.path.stat().st_ino != self._inode:
            self._open()
        if not self._f:
            self._open()
            return []
        lines = []
        while True:
            line = self._f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                lines.append(line)
        return lines


def run_realtime(eve_path, xgb_model, xgb_scaler, xgb_threshold,
                 inv_class_map, class_map, is_multiclass,
                 lstm_model, lstm_scaler, lstm_meta, output_path):
    tailer     = EveJsonTailer(eve_path)
    flow_cache = FlowEnrichment()      # http/tls/dns onbellegi
    ip_buffer  = IPBuffer()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  GERCEK ZAMANLI HIBRIT MOD v2")
    print(f"  Eve.json : {eve_path}")
    print(f"  Cikti    : {output_path}")
    print(f"  [Calisiyor -- Ctrl+C ile dur]\n")

    stats        = {"total": 0, "attack": 0, "benign": 0, "skip": 0}
    attack_types = defaultdict(int)
    start_time   = time.time()

    with open(output_path, "a", encoding="utf-8") as out:
        try:
            while True:
                for line in tailer.readlines():
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue

                    etype = event.get("event_type")

                    # http/tls/dns → onbellege al
                    if etype in ("http", "tls", "dns"):
                        flow_cache.ingest(event)
                        continue

                    if etype != "flow":
                        continue

                    stats["total"] += 1
                    enrichment = flow_cache.get(event.get("flow_id"))

                    alert = hybrid_predict(
                        event, enrichment,
                        xgb_model, xgb_scaler, xgb_threshold,
                        inv_class_map, class_map, is_multiclass,
                        lstm_model, lstm_scaler, lstm_meta,
                        ip_buffer,
                    )

                    if alert:
                        stats["attack"] += 1
                        lbl = alert["ai"]["label"]
                        attack_types[lbl] += 1
                        out.write(json.dumps(alert) + "\n")
                        out.flush()
                        print(
                            f"[ALERT] {alert['ai']['stage']:<10} "
                            f"{alert['src_ip']:18} -> {alert['dest_ip']:18} "
                            f"{lbl:<14} "
                            f"conf={alert['ai']['confidence']:.3f}"
                        )
                    else:
                        stats["benign"] += 1

                    if stats["total"] % 10_000 == 0:
                        elapsed = time.time() - start_time
                        fps     = stats["total"] / max(elapsed, 0.001)
                        print(f"[STAT] total={stats['total']:,} "
                              f"attack={stats['attack']:,} "
                              f"({fps:,.0f} flow/s)")

                time.sleep(POLL_S)

        except KeyboardInterrupt:
            print(f"\n\n  [Durduruldu]")
            print(f"  Toplam : {stats['total']:,} flow")
            print(f"  Alert  : {stats['attack']:,}")
            if attack_types:
                print(f"  Turler :")
                for t, c in sorted(attack_types.items(), key=lambda x: -x[1]):
                    print(f"    {t:<16}: {c:,}")


# ============================================================================
# ANA
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hibrit IDS v2 - XGBoost + LSTM Cikarim Motoru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  Canli:   python3 hybrid_inference.py
  Batch:   python3 hybrid_inference.py --eve test.json --batch
  Elle:    python3 hybrid_inference.py \\
               --xgb_model ids_model_v6_final.pkl \\
               --lstm_model lstm_best.keras
        """,
    )
    parser.add_argument("--eve",         default=str(DEFAULT_EVE))
    parser.add_argument("--output",      default=str(DEFAULT_OUT))
    parser.add_argument("--xgb_model",   default=str(DEFAULT_XGB_MODEL))
    parser.add_argument("--lstm_model",  default=str(DEFAULT_LSTM_MODEL))
    parser.add_argument("--lstm_scaler", default=str(DEFAULT_LSTM_SCALER))
    parser.add_argument("--lstm_meta",   default=str(DEFAULT_LSTM_META))
    parser.add_argument("--threshold",   type=float, default=None)
    parser.add_argument("--batch",       action="store_true")
    args = parser.parse_args()

    # XGBoost yukle
    xgb_model, xgb_scaler, xgb_threshold, inv_cmap, cmap, is_multiclass = \
        load_xgboost(Path(args.xgb_model))
    if args.threshold is not None:
        xgb_threshold = args.threshold
    print(f"  XGB Threshold : {xgb_threshold}")
    print(f"  DoS/DDoS skip : sinif {sorted(XGB_DOS_DDOS_CLASSES)} → LSTM atlanir")

    # LSTM yukle
    lstm_model, lstm_scaler, lstm_meta = load_lstm(
        Path(args.lstm_model), Path(args.lstm_scaler), Path(args.lstm_meta)
    )

    print(f"\n  LSTM etiketler : {LSTM_INV_CLASS_MAP}")
    print(f"  Pencere boyutu : {WINDOW_SIZE} akis")

    if args.batch:
        run_batch(Path(args.eve), xgb_model, xgb_scaler, xgb_threshold,
                  inv_cmap, cmap, is_multiclass,
                  lstm_model, lstm_scaler, lstm_meta, Path(args.output))
    else:
        run_realtime(Path(args.eve), xgb_model, xgb_scaler, xgb_threshold,
                     inv_cmap, cmap, is_multiclass,
                     lstm_model, lstm_scaler, lstm_meta, Path(args.output))
