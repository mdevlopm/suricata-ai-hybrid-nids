#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_hybrid_e2e_attacks.py
Runs each attack validation file through hybrid_inference.py (default settings)
and computes the end-to-end recall for each attack type.
"""

import sys
import subprocess
import json
from pathlib import Path

BASE_DIR = Path("/run/media/mehmet/siber data1/ai modeli xgboost")
HYBRID_SCRIPT = BASE_DIR / "pipeline/hybrid_inference.py"

attack_files = {
    "DoS": BASE_DIR / "data/eve/validation_DoS_10k.json",
    "DDoS (CICIDS2018)": BASE_DIR / "data/eve/validation_DDoS_10k.json",
    "WebAttack": BASE_DIR / "data/eve/validation_WebAttack_10k.json",
    "Bot (Cleaned)": BASE_DIR / "data/eve/validation_Bot_10k.json",
    "Infiltration": BASE_DIR / "data/eve/validation_Infiltration_10k.json",
}

results = {}

for atype, fpath in attack_files.items():
    out_json = Path(f"/tmp/alerts_{atype.replace(' ', '_')}.json")
    cmd = [
        "python3", str(HYBRID_SCRIPT),
        "--eve", str(fpath),
        "--batch",
        "--output", str(out_json)
    ]
    print(f"Running E2E test for {atype}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Count total flows and alert lines
    total_flows = 0
    with open(fpath) as f:
        total_flows = sum(1 for line in f if line.strip())
        
    alert_flows = 0
    if out_json.exists():
        with open(out_json) as f:
            alert_flows = sum(1 for line in f if line.strip())
            
    rec = (alert_flows / max(total_flows, 1)) * 100.0
    results[atype] = (alert_flows, total_flows, rec)
    print(f"  • {atype:<20}: {alert_flows:,} / {total_flows:,} detected ({rec:5.2f}%)")

print("\n" + "="*80)
print("FINAL END-TO-END ATTACK RECALL SUMMARY (via hybrid_inference.py default)")
print("="*80)
for atype, (det, tot, rec) in results.items():
    print(f"  {atype:<20}: {det:>6,} / {tot:>6,} ({rec:>6.2f}%)")
print("="*80)
