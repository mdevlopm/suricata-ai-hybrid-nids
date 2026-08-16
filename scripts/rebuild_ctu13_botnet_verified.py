#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verified CTU-13 Relabeler with Exact Timezone Synchronization (Prague UTC+2 CEST vs EVE UTC+3)
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

TZ_PRAGUE_SUMMER = timezone(timedelta(hours=2)) # CTU Prague CEST (UTC+2)

def parse_binetflow_ts(ts_str: str) -> float:
    """Parse binetflow timestamp 'YYYY/MM/DD HH:MM:SS.mmmmmm' in Prague CEST (UTC+2)."""
    ts_str = str(ts_str).strip()
    try:
        dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
        return dt.replace(tzinfo=TZ_PRAGUE_SUMMER).timestamp()
    except Exception:
        try:
            dt = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            return dt.replace(tzinfo=TZ_PRAGUE_SUMMER).timestamp()
        except Exception:
            return 0.0

def parse_eve_ts(ts_str: str) -> float:
    """Parse Suricata EVE ISO timestamp with timezone."""
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
    base_dir = Path("/run/media/mehmet/siber data1/ai modeli xgboost")
    ctu_dir = base_dir / "data/CTU-13"
    eve_dir = base_dir / "data/eve/CTU-13"
    out_dir = base_dir / "data/relabeled_ctu13"
    out_dir.mkdir(parents=True, exist_ok=True)

    comb_path = out_dir / "eve_Bot.json"
    all_comb_lines = []
    report = []
    start_all = time.time()

    for sc_dir in sorted(ctu_dir.glob("scenario_*")):
        sc_name = sc_dir.name
        eve_path = eve_dir / sc_name / "eve.json"
        
        binet_files = list(sc_dir.glob("*.binetflow")) + list(sc_dir.glob("**/*.binetflow"))
        if not eve_path.exists() or not binet_files:
            continue
            
        binet_file = binet_files[0]
        t0 = time.time()
        idx = defaultdict(list)
        
        with open(binet_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("StartTime") or line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) < 15:
                    continue
                ts_ep = parse_binetflow_ts(parts[0])
                if ts_ep <= 0:
                    continue
                proto = parts[2].strip()
                saddr = parts[3].strip()
                sport = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                daddr = parts[6].strip()
                dport = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
                lbl = parts[14].strip()
                k = make_key(saddr, sport, daddr, dport, proto)
                idx[k].append((ts_ep, lbl))
                
        sc_out = out_dir / sc_name
        sc_out.mkdir(parents=True, exist_ok=True)
        sc_bot_path = sc_out / "eve_Bot.json"
        
        s_tot, s_mat, s_bot = 0, 0, 0
        sc_lines = []
        
        with open(eve_path, "r", encoding="utf-8", errors="replace") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event_type") != "flow":
                    continue
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
                                ev["ctu_scenario"] = sc_name
                                l_str = json.dumps(ev) + "\n"
                                sc_lines.append(l_str)
                                all_comb_lines.append(l_str)
                                s_bot += 1
                                
        with open(sc_bot_path, "w", encoding="utf-8") as sf:
            sf.writelines(sc_lines)
            sf.flush()
            os.fsync(sf.fileno())
            
        pct = (s_mat / max(s_tot, 1)) * 100
        report.append((sc_name, s_tot, s_mat, pct, s_bot))
        print(f"[{sc_name}] Suricata Flows={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Flows Saved={s_bot:,} ({time.time()-t0:.1f}s)")

    with open(comb_path, "w", encoding="utf-8") as cf:
        cf.writelines(all_comb_lines)
        cf.flush()
        os.fsync(cf.fileno())

    dur = time.time() - start_all
    tot_sur = sum(r[1] for r in report)
    tot_mat = sum(r[2] for r in report)
    tot_bot = sum(r[4] for r in report)
    print("=" * 80)
    print(f"CTU-13 BOTNET RELABELING COMPLETE in {dur:.1f}s")
    print(f"Total Suricata Flows   : {tot_sur:,}")
    print(f"Total Matched Flows    : {tot_mat:,} ({(tot_mat/max(tot_sur,1))*100:.2f}%)")
    print(f"Total Botnet Flows Saved : {tot_bot:,} -> {comb_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
