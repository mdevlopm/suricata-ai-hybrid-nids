#!/usr/bin/env python3
"""
CICIDS2018 Hash‑Map Relabeling Tool
Milyonlarca satır CSV ile EVE'yi O(n+m) zamanda eşleştirir.

Kullanım:
    python3 relabel_cicids2018.py \
        --csv Thursday-15-02-2018.csv \
        --eve Thursday-15-02-2018/eve.json \
        --out ./relabeled
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# ════════════════ Protokol / Etiket Dönüşümleri ════════════════
def csv_proto_to_eve(proto):
    """CSV’deki sayısal protokol kodunu EVE stringine çevirir."""
    p = str(proto).strip().upper()
    if p in ("6", "TCP"):
        return "TCP"
    if p in ("17", "UDP"):
        return "UDP"
    if p in ("1", "ICMP"):
        return "ICMP"
    return p

def normalize_label(raw_label):
    """Uzun etiket adını ana sınıfa indirger."""
    label = str(raw_label).strip()
    ul = label.upper()
    if ul == "BENIGN":
        return "Benign"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("DDoS"):
        return "DDoS"
    if "Web Attack" in label or "Web attack" in label:
        return "WebAttack"
    if label.startswith("Infiltration"):
        return "Infiltration"
    if label.startswith("Bot"):
        return "Bot"
    if "Patator" in label:
        return "BruteForce"
    return label

# ════════════════ Zaman Çözümleme ════════════════
def parse_csv_timestamp(ts):
    """CICIDS2018 CSV timestamp'lerini datetime nesnesine çevirir."""
    ts = str(ts).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Tanınamayan CSV timestamp: {ts}")

def parse_eve_timestamp(ts_str):
    """EVE ISO 8601 → datetime."""
    ts_str = str(ts_str).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str)

# ════════════════ Ana İş Akışı ════════════════
def build_csv_hash(csv_path):
    """CSV'yi pandas DataFrame olarak okuyup hash‑map oluşturur."""
    df = pd.read_csv(csv_path, dtype=str)
    
    col_map = {
        'dst_port': ['Dst Port', 'Destination Port', 'dst_port'],
        'proto': ['Protocol', 'proto'],
        'timestamp': ['Timestamp', 'timestamp'],
        'label': ['Label', 'label'],
    }
    opt_map = {
        'src_ip': ['Src IP', 'Source IP', 'src_ip'],
        'src_port': ['Src Port', 'Source Port', 'src_port'],
        'dst_ip': ['Dst IP', 'Destination IP', 'dst_ip'],
    }
    
    cols = {}
    for key, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                cols[key] = c
                break
        if key not in cols:
            raise KeyError(f"CSV'de {key} sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")
            
    has_full_5tuple = True
    for key, candidates in opt_map.items():
        found = False
        for c in candidates:
            if c in df.columns:
                cols[key] = c
                found = True
                break
        if not found:
            has_full_5tuple = False

    hash_map = defaultdict(list)
    for _, row in df.iterrows():
        try:
            dst_port = int(row[cols['dst_port']])
            proto    = csv_proto_to_eve(row[cols['proto']])
            ts       = parse_csv_timestamp(row[cols['timestamp']])
            label    = normalize_label(row[cols['label']])
            
            if has_full_5tuple:
                src_ip   = str(row[cols['src_ip']]).strip()
                src_port = int(row[cols['src_port']])
                dst_ip   = str(row[cols['dst_ip']]).strip()
                key = (src_ip, src_port, dst_ip, dst_port, proto)
            else:
                key = (dst_port, proto)
        except (ValueError, KeyError):
            continue
            
        hash_map[key].append((ts, label))

    return hash_map, len(df), has_full_5tuple

def relabel(eve_path, hash_data, out_dir):
    csv_hash, _, has_full_5tuple = hash_data
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_handles = {}
    unmatched_f = open(out_dir / "unmatched.json", "w", encoding="utf-8")

    matched = 0
    unmatched = 0
    total_flows = 0

    with open(eve_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "flow":
                continue

            total_flows += 1
            src_ip   = event.get("src_ip", "")
            src_port = event.get("src_port", 0)
            dst_ip   = event.get("dest_ip", "")
            dst_port = event.get("dest_port", 0)
            proto    = (event.get("proto") or "").upper()

            flow = event.get("flow", {})
            ts_str  = flow.get("start") or event.get("timestamp", "")
            try:
                ev_ts = parse_eve_timestamp(ts_str)
            except:
                ev_ts = None

            if has_full_5tuple:
                key = (src_ip, src_port, dst_ip, dst_port, proto)
            else:
                key = (dst_port, proto)

            candidates = csv_hash.get(key)
            if not candidates:
                unmatched += 1
                unmatched_f.write(json.dumps(event) + "\n")
                continue

            # En iyi eşleşmeyi bul
            best_label = None
            if ev_ts:
                best_diff = float('inf')
                for csv_ts, label in candidates:
                    raw_diff = abs((ev_ts - csv_ts).total_seconds())
                    # Saat cinsinden farkı mod 3600 alarak zaman dilimi sapmasını (timezone shift) kompanse et
                    diff = raw_diff % 3600
                    if diff > 1800:
                        diff = 3600 - diff
                    if diff <= 120.0 and diff < best_diff:
                        best_diff = diff
                        best_label = label
                if best_label is None:
                    # Zaman aralığında bulamazsa ilk eşleşen adayı al
                    best_label = candidates[0][1]
            else:
                best_label = candidates[0][1]

            # Etiketli event'i yaz
            event["true_label"] = best_label
            if best_label not in file_handles:
                file_handles[best_label] = open(out_dir / f"eve_{best_label}.json", "w", encoding="utf-8")
            file_handles[best_label].write(json.dumps(event) + "\n")
            matched += 1

    for fh in file_handles.values():
        fh.close()
    unmatched_f.close()

    return total_flows, matched, unmatched

# ════════════════ CLI ════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CICIDS2018 Hash‑Map Relabeling")
    parser.add_argument("--csv", required=True, help="CSV dosyası")
    parser.add_argument("--eve", required=True, help="EVE JSON dosyası")
    parser.add_argument("--out", required=True, help="Çıktı dizini")
    args = parser.parse_args()

    print("[1/3] CSV hash‑map oluşturuluyor...")
    csv_hash, csv_rows = build_csv_hash(args.csv)
    print(f"       {csv_rows:,} CSV satırı işlendi.")

    print("[2/3] EVE eşleştiriliyor...")
    total, matched, unmatched = relabel(args.eve, csv_hash, args.out)

    print(f"\n{'='*45}")
    print(f"Toplam EVE akışı: {total:>12,}")
    print(f"Eşleşen (etiketli): {matched:>12,}  (%{100*matched/total:.1f})")
    print(f"Eşleşmeyen:         {unmatched:>12,}  (%{100*unmatched/total:.1f})")
    print(f"Çıktı dizini:       {Path(args.out).resolve()}")
