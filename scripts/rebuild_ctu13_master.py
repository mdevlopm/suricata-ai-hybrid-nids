#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master CTU-13 Relabeling Script:
1. Loads all 13 CTU-13 capture*.binetflow files.
2. Parses timestamps in Prague CEST (UTC+2).
3. Matches every eve.json in data/eve/CTU-13/ against the ground truth.
4. Generates the clean, complete data/relabeled_ctu13/eve_Bot.json.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7

TZ_PRAGUE_SUMMER = timezone(timedelta(hours=2))

def parse_binetflow_ts(ts_str: str) -> float:
    ts_str = str(ts_str).strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).replace(tzinfo=TZ_PRAGUE_SUMMER).timestamp()
        except Exception:
            continue
    return 0.0

def parse_eve_ts(ts_str: str) -> float:
    if not ts_str: return 0.0
    ts_str = str(ts_str).strip()
    try:
        from dateutil.parser import isoparse
        return isoparse(ts_str).timestamp()
    except Exception:
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

def norm_p(p: str) -> str:
    p = str(p).lower().strip()
    if p in ('tcp', '6'): return 'tcp'
    if p in ('udp', '17'): return 'udp'
    if p in ('icmp', '1'): return 'icmp'
    return p

def is_bot(lbl: str) -> bool:
    if not lbl: return False
    l = lbl.lower()
    return 'botnet' in l or 'c&c' in l or 'cc' in l or 'bot' in l

def make_key(ip1, port1, ip2, port2, proto):
    p1 = (str(ip1).strip(), int(port1 or 0))
    p2 = (str(ip2).strip(), int(port2 or 0))
    return (p1, p2, norm_p(proto)) if p1 <= p2 else (p2, p1, norm_p(proto))

def main():
    base_dir = Path(__file__).resolve().parent.parent
    ctu_dir = base_dir / "data/CTU-13"
    eve_dir = base_dir / "data/eve/CTU-13"
    out_dir = base_dir / "data/relabeled_ctu13"
    out_dir.mkdir(parents=True, exist_ok=True)

    comb_path = out_dir / "eve_Bot.json"
    all_comb_lines = []
    
    binet_files = sorted(ctu_dir.glob("*/*.binetflow"))
    eve_files = sorted(eve_dir.glob("*/eve.json"))

    print("=" * 80)
    print("MASTER CTU-13 GROUND TRUTH BOTNET EXTRACTION")
    print(f"Found {len(binet_files)} binetflow files and {len(eve_files)} eve.json files.")
    print("=" * 80)

    # For each eve file, determine which binetflow file(s) cover that date
    total_suricata = 0
    total_matched = 0
    total_bot_saved = 0
    start_time = time.time()

    for eve_path in eve_files:
        # Determine capture date from first flow
        first_date = None
        with open(eve_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try: ev = json.loads(line)
                except: continue
                if ev.get("event_type") == "flow":
                    st = ev.get("flow", {}).get("start") or ev.get("timestamp", "")
                    first_date = st[:10].replace("-", "") # e.g. "20110810"
                    break
        
        if not first_date:
            continue

        # Find matching binetflow files with same date string
        matching_binets = [bf for bf in binet_files if first_date in bf.name]
        if not matching_binets:
            # Fallback: if no date match, use all binetflow files in parent
            matching_binets = binet_files

        print(f"\nProcessing {eve_path.parent.name} (Date={first_date})...")
        print(f"  Matching binetflow files: {[b.name for b in matching_binets]}")

        # Index matching binetflow files
        idx = defaultdict(list)
        for bf in matching_binets:
            with open(bf, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("StartTime") or line.startswith("#") or not line.strip(): continue
                    parts = line.strip().split(",")
                    if len(parts) < 15: continue
                    ts_ep = parse_binetflow_ts(parts[0])
                    if ts_ep <= 0: continue
                    proto = parts[2].strip()
                    saddr = parts[3].strip()
                    sport = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                    daddr = parts[6].strip()
                    dport = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
                    lbl = parts[14].strip()
                    k = make_key(saddr, sport, daddr, dport, proto)
                    idx[k].append((ts_ep, lbl))

        s_tot, s_mat, s_bot = 0, 0, 0
        sc_lines = []

        with open(eve_path, "r", encoding="utf-8", errors="replace") as in_f:
            for line in in_f:
                line = line.strip()
                if not line: continue
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("event_type") != "flow": continue
                s_tot += 1

                flow = ev.get("flow", {})
                sip = ev.get("src_ip", "")
                sport = int(ev.get("src_port", 0) or 0)
                dip = ev.get("dest_ip", "")
                dport = int(ev.get("dest_port", 0) or 0)
                proto = ev.get("proto", "tcp")
                ts_str = flow.get("start") or ev.get("timestamp", "")
                ts_ep = parse_eve_ts(ts_str)

                k = make_key(sip, sport, dip, dport, proto)
                cands = idx.get(k)
                if cands:
                    best_lbl = None
                    min_diff = float("inf")
                    for (g_ts, g_lbl) in cands:
                        diff = abs(ts_ep - g_ts)
                        if diff <= 5.0 and diff < min_diff:
                            min_diff = diff
                            best_lbl = g_lbl
                    if best_lbl is not None:
                        s_mat += 1
                        if is_bot(best_lbl):
                            feat = extract_features_v7(ev)
                            if feat is not None:
                                ev["gt_label"] = best_lbl
                                ev["ctu_source_eve"] = eve_path.parent.name
                                l_str = json.dumps(ev) + "\n"
                                sc_lines.append(l_str)
                                all_comb_lines.append(l_str)
                                s_bot += 1

        sc_out = out_dir / eve_path.parent.name
        sc_out.mkdir(parents=True, exist_ok=True)
        with open(sc_out / "eve_Bot.json", "w", encoding="utf-8") as sf:
            sf.writelines(sc_lines)

        pct = (s_mat / max(s_tot, 1)) * 100
        print(f"  -> Suricata Flows: {s_tot:,} | Matched: {s_mat:,} ({pct:.2f}%) | Botnet Flows Saved: {s_bot:,}")
        total_suricata += s_tot
        total_matched += s_mat
        total_bot_saved += s_bot

    with open(comb_path, "w", encoding="utf-8") as cf:
        cf.writelines(all_comb_lines)
        cf.flush()
        os.fsync(cf.fileno())

    print("\n" + "=" * 80)
    print(f"CTU-13 MASTER EXTRACTION COMPLETE in {time.time() - start_time:.1f}s")
    print(f"Total Suricata Flows Processed : {total_suricata:,}")
    print(f"Total Matched Flows            : {total_matched:,} ({(total_matched/max(total_suricata,1))*100:.2f}%)")
    print(f"Total Verified Botnet Flows    : {total_bot_saved:,} -> {comb_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
