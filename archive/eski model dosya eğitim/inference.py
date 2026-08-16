# -*- coding: utf-8 -*-
"""
inference.py - Raspberry Pi'de calisir
========================================
Iki mod:

  1. Gercek zamanli (varsayilan):
     eve.json'u tail -f gibi izler, yeni flow geldikce tahmin eder.
     python3 inference.py --eve /var/log/suricata/eve.json

  2. Batch modu (--batch):
     Mevcut eve.json'u bastan sonuna isler, biter, cikar.
     Gecmis trafigi test etmek icin kullan.
     python3 inference.py --eve test_deneme/eve.json --batch
"""

import json, time, pickle, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

DEFAULT_EVE    = Path("/var/log/suricata/eve.json")
DEFAULT_MODEL  = Path("/opt/ids/ids_model.pkl")
DEFAULT_OUTPUT = Path("/var/log/suricata/ai_alerts.json")
POLL_S         = 0.5

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE UZAYI -- egitim.py ile BIREBIR AYNI
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "duration_s", "pkts_toserver", "pkts_toclient",
    "bytes_toserver", "bytes_toclient", "total_pkts", "total_bytes",
    "pkt_rate", "byte_rate", "bytes_per_pkt",
    "upload_byte_ratio", "upload_pkt_ratio", "byte_asymmetry",
    "dest_port", "src_port", "is_well_known_port", "is_high_port",
]
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(event: dict):
    if event.get("event_type") != "flow":
        return None, None

    flow = event.get("flow", {})
    pts  = float(flow.get("pkts_toserver",  0) or 0)
    ptc  = float(flow.get("pkts_toclient",  0) or 0)
    bts  = float(flow.get("bytes_toserver", 0) or 0)
    btc  = float(flow.get("bytes_toclient", 0) or 0)

    try:
        t0 = datetime.fromisoformat((flow.get("start") or event.get("timestamp","")).replace("Z","+00:00"))
        t1 = datetime.fromisoformat((flow.get("end") or "").replace("Z","+00:00"))
        dur = max((t1-t0).total_seconds(), 0.0)
    except:
        dur = 0.0

    dp = int(event.get("dest_port", 0) or 0)
    sp = int(event.get("src_port",  0) or 0)
    tp = pts+ptc; tb = bts+btc
    sd = max(dur,1e-6); spk = max(tp,1); sb = max(tb,1)

    features = np.array([[
        dur, pts, ptc, bts, btc, tp, tb,
        tp/sd, tb/sd, tb/spk,
        bts/sb, pts/spk, abs(bts-btc)/sb,
        dp, sp, int(dp<1024), int(dp>=49152),
    ]], dtype=np.float32)

    meta = {
        "timestamp"  : event.get("timestamp",""),
        "src_ip"     : event.get("src_ip",""),
        "dest_ip"    : event.get("dest_ip",""),
        "src_port"   : sp,
        "dest_port"  : dp,
        "proto"      : event.get("proto",""),
        "flow_id"    : event.get("flow_id",""),
        "duration_s" : round(dur, 3),
        "total_bytes": int(tb),
        "byte_rate"  : round(tb/sd, 1),
    }
    return features, meta


def load_model(path: Path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    print(f"Model    : {path}")
    print(f"  Egitim   : {b.get('trained_on','?')}")
    print(f"  Threshold: {b.get('threshold', 0.5)}")
    m = b.get("metrics", {})
    if m:
        print(f"  Accuracy : {m.get('accuracy',0)*100:.2f}%")
        print(f"  FAR      : {m.get('far',0)*100:.3f}%")
        print(f"  Recall   : {m.get('recall',0)*100:.2f}%")
    saved = b.get("feature_cols", [])
    if saved != FEATURE_COLS:
        print("\n  UYARI: feature_cols uyusmuyor!")
    return b["model"], b["scaler"], b.get("threshold", 0.5), b.get("inv_class_map", {})


def process_event(event, model, scaler, threshold, inv_class_map):
    """Tek bir event'i isle, (alert_dict veya None) dondur."""
    features, meta = extract_features(event)
    if features is None:
        return None

    prob     = model.predict_proba(scaler.transform(features))[0]
    prob_atk = 1 - prob[0]
    pred_cls = int(prob.argmax())
    label    = inv_class_map.get(pred_cls, str(pred_cls))

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
        "flow_id"   : meta["flow_id"],
        "ai": {
            "label"      : label,
            "confidence" : round(float(prob_atk), 4),
            "duration_s" : meta["duration_s"],
            "total_bytes": meta["total_bytes"],
            "byte_rate"  : meta["byte_rate"],
        }
    }


def run_batch(eve_path: Path, model, scaler, threshold, inv_class_map, output_path: Path):
    """
    Batch mod: dosyayi bastan sonuna isle, biter, cikar.
    Gecmis eve.json'u test etmek icin.
    """
    print(f"\nBATCH MOD")
    print(f"Eve.json : {eve_path}")
    print(f"Cikti    : {output_path}\n")

    stats = {"total":0, "attack":0, "benign":0, "skip":0}
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
                    print(f"  [ilerleme] {stats['total']:,} flow islendi...")

    # Ozet
    print(f"\n{'='*55}")
    print("BATCH SONUCU:")
    print(f"  Toplam flow : {stats['total']:,}")
    print(f"  Benign      : {stats['benign']:,}")
    print(f"  Alert       : {stats['attack']:,}  ({stats['attack']/max(stats['total'],1)*100:.1f}%)")
    if attack_types:
        print(f"\n  Saldiri turleri:")
        for t, c in sorted(attack_types.items(), key=lambda x: -x[1]):
            print(f"    {t:<15}: {c:,}")
    print(f"\n  Kaydedildi  : {output_path}")
    print(f"{'='*55}")


class EveJsonTailer:
    def __init__(self, path):
        self.path = path
        self._f = self._inode = None
        self._open()

    def _open(self):
        if self._f: self._f.close()
        if self.path.exists():
            self._f = open(self.path, encoding="utf-8", errors="replace")
            self._f.seek(0, 2)
            self._inode = self.path.stat().st_ino
        else:
            self._f = self._inode = None

    def readlines(self):
        if not self.path.exists():
            if self._f: self._f.close(); self._f = None
            return []
        if self.path.stat().st_ino != self._inode:
            self._open()
        if not self._f:
            self._open(); return []
        lines = []
        while True:
            line = self._f.readline()
            if not line: break
            line = line.strip()
            if line: lines.append(line)
        return lines


def run_realtime(eve_path: Path, model, scaler, threshold, inv_class_map, output_path: Path):
    """Gercek zamanli mod: yeni flow'lari surekli izle."""
    tailer = EveJsonTailer(eve_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGERCEK ZAMANLI MOD")
    print(f"Eve.json : {eve_path}")
    print(f"Cikti    : {output_path}")
    print("[Calisıyor -- Ctrl+C ile dur]\n")

    stats = {"total":0, "attack":0, "benign":0}

    with open(output_path, "a", encoding="utf-8") as out:
        while True:
            for line in tailer.readlines():
                try: event = json.loads(line)
                except: continue

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
                    print(f"[STAT] total={stats['total']} attack={stats['attack']} benign={stats['benign']}")

            time.sleep(POLL_S)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eve",       default=str(DEFAULT_EVE))
    p.add_argument("--model",     default=str(DEFAULT_MODEL))
    p.add_argument("--output",    default=str(DEFAULT_OUTPUT))
    p.add_argument("--threshold", default=None, type=float)
    p.add_argument("--batch",     action="store_true", help="Dosyayi bastan isle ve cik")
    args = p.parse_args()

    model, scaler, threshold, inv_class_map = load_model(Path(args.model))
    if args.threshold:
        threshold = args.threshold
    print(f"Threshold: {threshold}")

    if args.batch:
        run_batch(Path(args.eve), model, scaler, threshold, inv_class_map, Path(args.output))
    else:
        run_realtime(Path(args.eve), model, scaler, threshold, inv_class_map, Path(args.output))
