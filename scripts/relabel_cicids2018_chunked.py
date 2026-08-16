#!/usr/bin/env python3
"""
CICIDS2018 Chunked Relabeling Tool
Processes large CSV files in chunks to avoid memory issues.
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
    p = str(proto).strip().upper()
    if p in ("6", "TCP"): return "TCP"
    if p in ("17", "UDP"): return "UDP"
    if p in ("1", "ICMP"): return "ICMP"
    return p

def normalize_label(raw_label):
    label = str(raw_label).strip()
    ul = label.upper()
    if ul == "BENIGN":
        return "Benign"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("DDoS") or label.startswith("DDOS"):
        return "DDoS"
    if "Web Attack" in label or "Web attack" in label:
        return "WebAttack"
    if label.startswith("Infiltration"):
        return "Infiltration"
    if label.startswith("Bot"):
        return "Bot"
    if "Patator" in label or "BruteForce" in label or "Bruteforce" in label:
        return "BruteForce"
    if "FTP" in label.upper() and "BRUTE" in label.upper():
        return "BruteForce"
    if "SSH" in label.upper() and "BRUTE" in label.upper():
        return "BruteForce"
    return label

def parse_csv_timestamp(ts):
    ts = str(ts).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Tanınamayan CSV timestamp: {ts}")

def parse_eve_timestamp(ts_str):
    ts_str = str(ts_str).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str)

# ════════════════ Chunked CSV Hash Building ════════════════
def build_csv_hash_chunked(csv_path, chunk_size=500_000):
    """Build hash map from CSV in chunks to avoid memory issues."""
    first_chunk = True
    hash_map = defaultdict(list)
    total_rows = 0
    has_full_5tuple = None
    
    cols_needed = ['Dst Port', 'Protocol', 'Timestamp', 'Label']
    opt_cols = ['Src IP', 'Src Port', 'Dst IP']
    
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunk_size, usecols=lambda c: c in cols_needed + opt_cols + ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Label', 'Flow ID']):
        if first_chunk:
            has_full_5tuple = all(c in chunk.columns for c in ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol'])
            first_chunk = False
        
        # Use only needed columns
        use_cols = ['Dst Port', 'Protocol', 'Timestamp', 'Label']
        if has_full_5tuple:
            use_cols = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Label']
        
        for _, row in chunk[use_cols].iterrows():
            try:
                if has_full_5tuple:
                    src_ip   = str(row['Src IP']).strip()
                    src_port = int(row['Src Port'])
                    dst_ip   = str(row['Dst IP']).strip()
                    dst_port = int(row['Dst Port'])
                    proto    = csv_proto_to_eve(row['Protocol'])
                    key = (src_ip, src_port, dst_ip, dst_port, proto)
                else:
                    dst_port = int(row['Dst Port'])
                    proto    = csv_proto_to_eve(row['Protocol'])
                    key = (dst_port, proto)
                
                ts    = parse_csv_timestamp(row['Timestamp'])
                label = normalize_label(row['Label'])
                
                hash_map[key].append((ts, label))
                total_rows += 1
            except (ValueError, KeyError):
                continue
        
        # Progress
        if total_rows % 1_000_000 == 0:
            print(f"  Processed {total_rows:,} rows...")
    
    print(f"  Total CSV rows processed: {total_rows:,} | 5-tuple: {has_full_5tuple}")
    return hash_map, total_rows, has_full_5tuple

def relabel_chunked(eve_path, hash_data, out_dir, batch_size=10000):
    csv_hash, _, has_full_5tuple = hash_data
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_handles = {}
    unmatched_f = open(out_dir / "unmatched.json", "w", encoding="utf-8")

    matched = 0
    unmatched = 0
    total_flows = 0
    batch_buffer = []

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

            best_label = None
            if ev_ts:
                best_diff = float('inf')
                for csv_ts, label in candidates:
                    raw_diff = abs((ev_ts - csv_ts).total_seconds())
                    diff = raw_diff % 3600
                    if diff > 1800:
                        diff = 3600 - diff
                    if diff <= 120.0 and diff < best_diff:
                        best_diff = diff
                        best_label = label
                if best_label is None:
                    best_label = candidates[0][1]
            else:
                best_label = candidates[0][1]

            event["true_label"] = best_label
            if best_label not in file_handles:
                file_handles[best_label] = open(out_dir / f"eve_{best_label}.json", "w", encoding="utf-8")
            file_handles[best_label].write(json.dumps(event) + "\n")
            matched += 1

            if total_flows % 100_000 == 0:
                print(f"  Processed {total_flows:,} EVE flows...")

    for fh in file_handles.values():
        fh.close()
    unmatched_f.close()

    return total_flows, matched, unmatched

# ════════════════ CLI ════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CICIDS2018 Chunked Relabeling")
    parser.add_argument("--csv", required=True, help="CSV dosyası")
    parser.add_argument("--eve", required=True, help="EVE JSON dosyası")
    parser.add_argument("--out", required=True, help="Çıktı dizini")
    parser.add_argument("--chunk-size", type=int, default=500000, help="CSV chunk size")
    args = parser.parse_args()

    print("[1/3] CSV hash-map oluşturuluyor (chunked)...")
    csv_hash, csv_rows, has_5tuple = build_csv_hash_chunked(args.csv, args.chunk_size)
    print(f"       {csv_rows:,} CSV satırı işlendi. 5-tuple: {has_5tuple}")

    print("[2/3] EVE eşleştiriliyor...")
    total, matched, unmatched = relabel_chunked(args.eve, (csv_hash, csv_rows, has_5tuple), args.out)

    print(f"\n{'='*45}")
    print(f"Toplam EVE akışı: {total:>12,}")
    print(f"Eşleşen (etiketli): {matched:>12,}  (%{100*matched/total:.1f})")
    print(f"Eşleşmeyen:         {unmatched:>12,}  (%{100*unmatched/total:.1f})")
    print(f"Çıktı dizini:       {Path(args.out).resolve()}")