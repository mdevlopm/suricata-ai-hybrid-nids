#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interfacev6.py - IDS v6 Nihai Canli Cikarim Motoru
====================================================
trainv6.py ile egitilmis XGBoost modelini kullanir.

Ozellikler:
  * Iki mod: gercek zamanli (tail) ve batch
  * 42 ozellik cikarimi (egitim ile birebir ayni)
  * Log rotasyonu takibi (inode)
  * Detayli periyodik raporlar
  * Bellek dostu (satir satir okuma)
  * Tum exceptionlar guvenli yakalanir

Kullanim:
  Gercek zamanli:
    python3 interfacev6.py --eve /var/log/suricata/eve.json

  Batch:
    python3 interfacev6.py --eve test.json --batch --output ./alerts.json

  Ozel esik ile:
    python3 interfacev6.py --eve /var/log/suricata/eve.json --threshold 0.65
"""

import argparse, json, pickle, time, sys
from pathlib import Path
from datetime import datetime
import numpy as np

# ============================================================================
# VARSAYILAN YOLLAR
# ============================================================================
DEFAULT_MODEL = Path("./ids_model_v6_final.pkl")
DEFAULT_EVE   = Path("/var/log/suricata/eve.json")
DEFAULT_OUT   = Path("/var/log/suricata/ai_alerts_v6.json")
POLL_S        = 0.5  # polling araligi (saniye)
REPORT_EVERY  = 100_000  # her N flow'da rapor


# ============================================================================
# OZELLIK CIKARIMI (trainv6.py ile birebir ayni)
# ============================================================================
def extract_features(event: dict):
    """Suricata flow event'inden tam 42 ozellik cikarir."""
    if event.get("event_type") != "flow":
        return None, None

    flow = event.get("flow", {})
    pts = float(flow.get("pkts_toserver",  0) or 0)
    ptc = float(flow.get("pkts_toclient",  0) or 0)
    bts = float(flow.get("bytes_toserver", 0) or 0)
    btc = float(flow.get("bytes_toclient", 0) or 0)

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

    features = np.array([[
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
    ]], dtype=np.float32)

    meta = {
        "timestamp"  : event.get("timestamp", ""),
        "src_ip"     : event.get("src_ip", ""),
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
# MODEL YUKLEME
# ============================================================================
def load_model(path: Path):
    """Egitim bundle'ini acar; model, scaler, threshold ve etiket haritasi doner."""
    with open(path, "rb") as f:
        b = pickle.load(f)

    print(f"\n  Model      : {path}")
    print(f"  Egitim     : {b.get('trained_on', '?')}")
    print(f"  Threshold  : {b.get('threshold', 0.5)}")
    print(f"  Features   : {b.get('n_features', '?')}")
    m = b.get("metrics", {})
    if m:
        print(f"  Accuracy   : {m.get('accuracy', 0) * 100:.2f}%")
        print(f"  Macro F1   : {m.get('macro_f1', 0) * 100:.2f}%")
        print(f"  FAR (opt)  : {m.get('far', 0) * 100:.3f}%")
        print(f"  Recall(opt): {m.get('recall', 0) * 100:.2f}%")
        print(f"  PR-AUC     : {m.get('pr_auc', 0):.4f}")

    return (
        b["model"],
        b["scaler"],
        b.get("threshold", 0.5),
        b.get("inv_class_map", {}),
        b.get("class_map", {}),
    )


# ============================================================================
# TEK BIR EVENT'I ISLE
# ============================================================================
def process_event(event, model, scaler, threshold, inv_class_map, class_map):
    """Flow event'ini siniflandirir, esik ustundeyse alert sozlugu dondurur."""
    features, meta = extract_features(event)
    if features is None:
        return None

    prob     = model.predict_proba(scaler.transform(features))[0]
    prob_atk = 1.0 - prob[0]  # Sinif 0 = Benign
    pred_cls = int(prob.argmax())
    label    = inv_class_map.get(pred_cls, str(pred_cls))

    if prob_atk < threshold:
        return None

    return {
        "timestamp"  : meta["timestamp"],
        "event_type" : "ai_alert_v6",
        "src_ip"     : meta["src_ip"],
        "dest_ip"    : meta["dest_ip"],
        "src_port"   : meta["src_port"],
        "dest_port"  : meta["dest_port"],
        "proto"      : meta["proto"],
        "app_proto"  : meta["app_proto"],
        "flow_id"    : meta["flow_id"],
        "ai": {
            "label"      : label,
            "confidence" : round(float(prob_atk), 4),
            "class_probs": {
                inv_class_map.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(prob)
            },
            "duration_s" : meta["duration_s"],
            "total_bytes": meta["total_bytes"],
            "byte_rate"  : meta["byte_rate"],
        }
    }


# ============================================================================
# BATCH MOD
# ============================================================================
def run_batch(eve_path, model, scaler, threshold, inv_class_map, class_map, output_path):
    """Dosyayi bastan sona islER, tamamlaninca cikar."""
    print(f"\n  BATCH MOD")
    print(f"  Eve.json : {eve_path}")
    print(f"  Cikti    : {output_path}\n")

    stats = {"total": 0, "attack": 0, "benign": 0, "skip": 0, "error": 0}
    attack_types = {}
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

                if event.get("event_type") != "flow":
                    stats["skip"] += 1
                    continue

                stats["total"] += 1
                alert = process_event(event, model, scaler, threshold,
                                       inv_class_map, class_map)
                if alert:
                    stats["attack"] += 1
                    lbl = alert["ai"]["label"]
                    attack_types[lbl] = attack_types.get(lbl, 0) + 1
                    out.write(json.dumps(alert) + "\n")
                else:
                    stats["benign"] += 1

                # Periyodik rapor
                if stats["total"] % REPORT_EVERY == 0:
                    elapsed = time.time() - start_time
                    fps = stats["total"] / max(elapsed, 0.001)
                    print(f"\n  --- RAPOR (her {REPORT_EVERY:,} flow) ---")
                    print(f"  Islenen    : {stats['total']:>12,}  ({fps:,.0f} flow/s)")
                    print(f"  Saldiri    : {stats['attack']:>12,}")
                    print(f"  Benign     : {stats['benign']:>12,}")
                    print(f"  Gecersiz   : {stats['skip']:>12,}")
                    print(f"  Sure       : {elapsed/60:>10.1f} dk")
                    if attack_types:
                        print(f"  --- Saldiri Turleri ---")
                        for lbl, cnt in sorted(attack_types.items(), key=lambda x: -x[1]):
                            print(f"    {lbl:<15}: {cnt:>10,}")
                    print()
                    out.flush()

    # Son ozet
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  BATCH SONUCU")
    print(f"{'='*60}")
    print(f"  Toplam flow : {stats['total']:>12,}")
    print(f"  Benign      : {stats['benign']:>12,}  ({stats['benign']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Alert       : {stats['attack']:>12,}  ({stats['attack']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Gecersiz    : {stats['skip']:>12,}")
    print(f"  Sure        : {elapsed/60:.1f} dk  ({stats['total']/max(elapsed,0.001):,.0f} flow/s)")
    if attack_types:
        print(f"\n  Saldiri Turleri:")
        for lbl, cnt in sorted(attack_types.items(), key=lambda x: -x[1]):
            print(f"    {lbl:<15}: {cnt:>10,}")
    print(f"\n  Kaydedildi  : {output_path}")
    print(f"{'='*60}")


# ============================================================================
# EVE.JSON TAILER (log rotasyonu destekli)
# ============================================================================
class EveJsonTailer:
    """Suricata eve.json dosyasini log rotasyonunu da takip ederek okur."""
    def __init__(self, path):
        self.path = path
        self._f = self._inode = None
        self._open()

    def _open(self):
        if self._f:
            self._f.close()
        if self.path.exists():
            self._f = open(self.path, encoding="utf-8", errors="replace")
            self._f.seek(0, 2)  # dosya sonuna git
            self._inode = self.path.stat().st_ino
        else:
            self._f = self._inode = None

    def readlines(self):
        if not self.path.exists():
            if self._f:
                self._f.close()
                self._f = None
            return []
        # Log rotasyonu: inode degisti mi?
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


# ============================================================================
# GERCEK ZAMANLI MOD
# ============================================================================
def run_realtime(eve_path, model, scaler, threshold, inv_class_map, class_map, output_path):
    """Surekli eve.json'u izler, yeni flow'lari siniflandirir."""
    tailer = EveJsonTailer(eve_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  GERCEK ZAMANLI MOD")
    print(f"  Eve.json : {eve_path}")
    print(f"  Cikti    : {output_path}")
    print(f"  Threshold: {threshold}")
    print(f"  [Calisiyor -- Ctrl+C ile dur]\n")

    stats = {"total": 0, "attack": 0, "benign": 0, "skip": 0}
    attack_types = {}
    start_time = time.time()

    with open(output_path, "a", encoding="utf-8") as out:
        try:
            while True:
                for line in tailer.readlines():
                    try:
                        event = json.loads(line)
                    except Exception:
                        stats["skip"] += 1
                        continue

                    if event.get("event_type") != "flow":
                        stats["skip"] += 1
                        continue

                    stats["total"] += 1
                    alert = process_event(event, model, scaler, threshold,
                                           inv_class_map, class_map)
                    if alert:
                        stats["attack"] += 1
                        lbl = alert["ai"]["label"]
                        attack_types[lbl] = attack_types.get(lbl, 0) + 1
                        out.write(json.dumps(alert) + "\n")
                        out.flush()
                    else:
                        stats["benign"] += 1

                    # Periyodik rapor
                    if stats["total"] % REPORT_EVERY == 0:
                        elapsed = time.time() - start_time
                        fps = stats["total"] / max(elapsed, 0.001)
                        print(f"\n  --- GERCEK ZAMANLI RAPOR ---")
                        print(f"  Islenen    : {stats['total']:>12,}  ({fps:,.0f} flow/s)")
                        print(f"  Saldiri    : {stats['attack']:>12,}")
                        print(f"  Benign     : {stats['benign']:>12,}")
                        print(f"  Sure       : {elapsed/60:>10.1f} dk")
                        if attack_types:
                            print(f"  --- Saldiri Turleri ---")
                            for lbl, cnt in sorted(attack_types.items(), key=lambda x: -x[1]):
                                print(f"    {lbl:<15}: {cnt:>10,}")
                        print()

                time.sleep(POLL_S)
        except KeyboardInterrupt:
            print("\n\n  [Durduruldu]")
            print(f"  Toplam: {stats['total']:,} flow islendi")
            print(f"  Alert : {stats['attack']:,} uretildi")


# ============================================================================
# ANA
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IDS AI v6 - XGBoost Nihai Cikarim Motoru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  Canli izleme:        python3 interfacev6.py
  Batch mod:           python3 interfacev6.py --eve test.json --batch
  Ozel threshold:      python3 interfacev6.py --threshold 0.65
  Farkli model:        python3 interfacev6.py --model /path/to/model.pkl
        """,
    )
    parser.add_argument("--eve",       default=str(DEFAULT_EVE),
                        help="eve.json yolu")
    parser.add_argument("--model",     default=str(DEFAULT_MODEL),
                        help="Model dosyasi (.pkl)")
    parser.add_argument("--output",    default=str(DEFAULT_OUT),
                        help="Alert cikti dosyasi")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Ozel esik degeri (varsayilan: egitimden)")
    parser.add_argument("--batch",     action="store_true",
                        help="Batch mod (tek seferlik isle ve cik)")
    args = parser.parse_args()

    model, scaler, threshold, inv_class_map, class_map = load_model(Path(args.model))
    if args.threshold is not None:
        threshold = args.threshold
    print(f"\n  Threshold: {threshold}")

    if args.batch:
        run_batch(Path(args.eve), model, scaler, threshold,
                  inv_class_map, class_map, Path(args.output))
    else:
        run_realtime(Path(args.eve), model, scaler, threshold,
                     inv_class_map, class_map, Path(args.output))
