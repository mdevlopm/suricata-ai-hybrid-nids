#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_relabel_batch.py - 4 Gerçek CSV + EVE JSON Setinde Relabel Çalıştırma ve Eşleşme Oranı Analizi
"""

import sys
import os
import time
from pathlib import Path

# Add scripts directory to import path
sys.path.insert(0, str(Path(__file__).parent))
from relabel_cicids2018 import build_csv_hash, relabel

tasks = [
    {
        "name": "Wednesday 14-02-2018 (Benign)",
        "csv": "data/csv/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
        "eve": "data/raw_pcap/Wednesday-14-02-2018/eve_Benign.json",
        "out": "data/relabeled/wed14"
    },
    {
        "name": "Thursday 15-02-2018 (DoS)",
        "csv": "data/csv/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
        "eve": "data/raw_pcap/Thursday-15-02-2018/eve_DoS.json",
        "out": "data/relabeled/thu15"
    },
    {
        "name": "Tuesday 20-02-2018 (DDoS)",
        "csv": "data/csv/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "eve": "data/raw_pcap/Tuesday-20-02-2018/eve_DDoS.json",
        "out": "data/relabeled/tue20"
    },
    {
        "name": "Wednesday 21-02-2018 (DDoS)",
        "csv": "data/csv/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
        "eve": "data/raw_pcap/Wednesday-21-02-2018/eve_DDoS.json",
        "out": "data/relabeled/wed21"
    }
]

print("==========================================================================")
print("CICIDS2018 RELABELING BATCH PROCESS STARTED")
print("==========================================================================")

results = []

for item in tasks:
    print(f"\n---> Running Relabel for: {item['name']}")
    start_t = time.time()
    
    print(f"     Loading CSV Hash: {item['csv']}")
    hash_data = build_csv_hash(item['csv'])
    csv_rows = hash_data[1]
    print(f"     CSV Rows Loaded: {csv_rows:,} (Full 5-tuple: {hash_data[2]})")
    
    print(f"     Matching EVE JSON: {item['eve']}")
    total, matched, unmatched = relabel(item['eve'], hash_data, item['out'])
    
    elapsed = time.time() - start_t
    match_pct = (matched / total * 100) if total > 0 else 0.0
    
    res = {
        "name": item["name"],
        "out_dir": str(Path(item["out"]).resolve()),
        "csv_rows": csv_rows,
        "total_eve": total,
        "matched": matched,
        "unmatched": unmatched,
        "match_pct": match_pct,
        "elapsed_s": elapsed
    }
    results.append(res)
    
    print(f"     DONE in {elapsed:.1f}s | EVE Flows: {total:,} | Matched: {matched:,} ({match_pct:.2f}%)")

print("\n" + "="*80)
print("RELABELING SUMMARY REPORT")
print("="*80)
print(f"{'Dataset / Day':<30} | {'EVE Flows':<12} | {'Matched':<12} | {'Match %':<10} | {'DeepSeek Est.':<12}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<30} | {r['total_eve']:<12,} | {r['matched']:<12,} | %{r['match_pct']:<9.2f} | %91.0 - %97.0")

print("-" * 80)
print(f"Output Base Directory: {Path('data/relabeled').resolve()}")
print("==========================================================================")
