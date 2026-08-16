"""
inference_v2.py - Raspberry Pi'de çalışır
==========================================
egitim_v2.py ile eğitilmiş modeli kullanır.
Feature extraction kodu birebir aynı.

Kullanım:
  Gerçek zamanlı:
    python3 inference_v2.py --eve /var/log/suricata/eve.json

  Batch:
    python3 inference_v2.py --eve test.json --batch
"""

import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

DEFAULT_EVE    = Path("/var/log/suricata/eve.json")
DEFAULT_MODEL  = Path("/opt/ids/ids_model_v2.pkl")
DEFAULT_OUTPUT = Path("/var/log/suricata/ai_alerts.json")
POLL_S         = 0.5


# ── FEATURE EXTRACTION (egitim_v2.py ile birebir aynı) ───────────────────────

def extract_features(event: dict):
    if event.get("event_type") != "flow":
        return None, None

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
    sd  = max(dur, 1e-6)
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
        float(app_proto not in ("http","dns","tls","dcerpc","smb","rdp","failed")),
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


# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────

def load_model(path: Path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    print(f"Model      : {path}")
    print(f"  Eğitim   : {b.get('trained_on', '?')}")
    print(f"  Threshold: {b.get('threshold', 0.5)}")
    print(f"  Features : {b.get('n_features', '?')}")
    m = b.get("metrics", {})
    if m:
        print(f"  Accuracy : {m.get('accuracy', 0) * 100:.2f}%")
        print(f"  FAR      : {m.get('far', 0) * 100:.3f}%")
        print(f"  Recall   : {m.get('recall', 0) * 100:.2f}%")
    return b["model"], b["scaler"], b.get("threshold", 0.5), b.get("inv_class_map", {})


# ── EVENT İŞLEME ─────────────────────────────────────────────────────────────

def process_event(event, model, scaler, threshold, inv_class_map):
    features, meta = extract_features(event)
    if features is None:
        return None

    prob       = model.predict_proba(scaler.transform(features))[0]
    prob_atk   = 1 - prob[0]
    pred_cls   = int(prob.argmax())
    label      = inv_class_map.get(pred_cls, str(pred_cls))

    if prob_atk < threshold:
        return None

    return {
        "timestamp" : meta["timestamp"],
        "event_type": "ai_alert",
        "src_ip"    : meta["src_ip"],
        "dest_ip"   : meta["dest_ip"],
        "src_port"  : meta["src_port"],
        "dest_port" : meta["dest_port"],
        "proto"     : meta["proto"],
        "app_proto" : meta["app_proto"],
        "flow_id"   : meta["flow_id"],
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


# ── BATCH MOD ─────────────────────────────────────────────────────────────────

def run_batch(eve_path, model, scaler, threshold, inv_class_map, output_path):
    print(f"\nBATCH MOD")
    print(f"Eve.json : {eve_path}")
    print(f"Çıktı    : {output_path}\n")

    stats = {"total": 0, "attack": 0, "benign": 0, "skip": 0}
    attack_types = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        with open(eve_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except:
                    stats["skip"] += 1
                    continue

                if event.get("event_type") != "flow":
                    stats["skip"] += 1
                    continue

                stats["total"] += 1
                alert = process_event(event, model, scaler, threshold, inv_class_map)

                if alert:
                    stats["attack"] += 1
                    lbl = alert["ai"]["label"]
                    attack_types[lbl] = attack_types.get(lbl, 0) + 1
                    out.write(json.dumps(alert) + "\n")
                    print(
                        f"  [ALERT] {alert['timestamp']} | "
                        f"{alert['src_ip']:20} -> {alert['dest_ip']:15}:{alert['dest_port']:5} | "
                        f"{lbl} | conf={alert['ai']['confidence']:.3f}"
                    )
                else:
                    stats["benign"] += 1

                if stats["total"] % 10000 == 0:
                    print(f"  [ilerleme] {stats['total']:,} flow işlendi...")

    print(f"\n{'=' * 55}")
    print("BATCH SONUCU:")
    print(f"  Toplam flow : {stats['total']:,}")
    print(f"  Benign      : {stats['benign']:,}")
    print(f"  Alert       : {stats['attack']:,}  ({stats['attack'] / max(stats['total'], 1) * 100:.1f}%)")
    if attack_types:
        print(f"\n  Saldırı türleri:")
        for t, c in sorted(attack_types.items(), key=lambda x: -x[1]):
            print(f"    {t:<15}: {c:,}")
    print(f"\n  Kaydedildi  : {output_path}")
    print(f"{'=' * 55}")


# ── GERÇEK ZAMANLI MOD ────────────────────────────────────────────────────────

class EveJsonTailer:
    def __init__(self, path):
        self.path = path
        self._f = self._inode = None
        self._open()

    def _open(self):
        if self._f:
            self._f.close()
        if self.path.exists():
            self._f = open(self.path, encoding="utf-8", errors="replace")
            self._f.seek(0, 2)
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


def run_realtime(eve_path, model, scaler, threshold, inv_class_map, output_path):
    tailer = EveJsonTailer(eve_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGERÇEK ZAMANLI MOD")
    print(f"Eve.json : {eve_path}")
    print(f"Çıktı    : {output_path}")
    print("[Çalışıyor -- Ctrl+C ile dur]\n")

    stats = {"total": 0, "attack": 0, "benign": 0}

    with open(output_path, "a", encoding="utf-8") as out:
        while True:
            for line in tailer.readlines():
                try:
                    event = json.loads(line)
                except:
                    continue

                if event.get("event_type") != "flow":
                    continue

                stats["total"] += 1
                alert = process_event(event, model, scaler, threshold, inv_class_map)

                if alert:
                    stats["attack"] += 1
                    out.write(json.dumps(alert) + "\n")
                    out.flush()
                    print(
                        f"[ALERT] {alert['timestamp']} | "
                        f"{alert['src_ip']} -> {alert['dest_ip']}:{alert['dest_port']} | "
                        f"{alert['ai']['label']} | conf={alert['ai']['confidence']:.3f}"
                    )
                else:
                    stats["benign"] += 1

                if stats["total"] % 500 == 0:
                    print(
                        f"[STAT] total={stats['total']} "
                        f"attack={stats['attack']} "
                        f"benign={stats['benign']}"
                    )

            time.sleep(POLL_S)


# ── ANA FONKSİYON ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eve",       default=str(DEFAULT_EVE))
    p.add_argument("--model",     default=str(DEFAULT_MODEL))
    p.add_argument("--output",    default=str(DEFAULT_OUTPUT))
    p.add_argument("--threshold", default=None, type=float)
    p.add_argument("--batch",     action="store_true")
    args = p.parse_args()

    model, scaler, threshold, inv_class_map = load_model(Path(args.model))
    if args.threshold:
        threshold = args.threshold
    print(f"Threshold: {threshold}")

    if args.batch:
        run_batch(
            Path(args.eve), model, scaler, threshold,
            inv_class_map, Path(args.output)
        )
    else:
        run_realtime(
            Path(args.eve), model, scaler, threshold,
            inv_class_map, Path(args.output)
        )
