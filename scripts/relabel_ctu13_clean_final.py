#!/usr/bin/env python3
import json
import time
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from dateutil.parser import isoparse

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7

def parse_ts(ts_str: str) -> float:
    if not ts_str:
        return 0.0
    ts_str = str(ts_str).strip()
    try:
        if "T" in ts_str:
            return isoparse(ts_str).timestamp()
        if "/" in ts_str or "-" in ts_str:
            parts = ts_str.split(".")
            main_ts = parts[0][:19]
            fmt = "%Y/%m/%d %H:%M:%S" if "/" in main_ts else "%Y-%m-%d %H:%M:%S"
            dt = datetime.strptime(main_ts, fmt)
            micros = 0.0
            if len(parts) > 1:
                m_digits = "".join([c for c in parts[1][:6] if c.isdigit()])
                if m_digits:
                    micros = float(f"0.{m_digits}")
            return dt.replace(tzinfo=timezone.utc).timestamp() + micros
    except Exception:
        pass
    return 0.0

def normalize_proto(p: str) -> str:
    p = str(p).lower().strip()
    if p in ("tcp", "6"): return "tcp"
    if p in ("udp", "17"): return "udp"
    if p in ("icmp", "1"): return "icmp"
    return p

def is_bot(lbl: str) -> bool:
    if not lbl: return False
    l = lbl.lower()
    return "botnet" in l or "c&c" in l or "cc" in l or "bot" in l

def process_scenario(sc_name: str, eve_path: Path, label_path: Path, out_dir: Path):
    index = defaultdict(list)
    t0 = time.time()
    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("StartTime") or line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 15:
                continue
            ts_epoch = parse_ts(parts[0])
            if ts_epoch <= 0:
                continue
            proto = normalize_proto(parts[2])
            saddr = parts[3].strip()
            sport = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
            daddr = parts[6].strip()
            dport = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
            lbl = parts[14].strip()
            if (saddr, sport) <= (daddr, dport):
                k = (saddr, sport, daddr, dport, proto)
            else:
                k = (daddr, dport, saddr, sport, proto)
            index[k].append((ts_epoch, lbl))
            
    sc_out_dir = out_dir / sc_name
    sc_out_dir.mkdir(parents=True, exist_ok=True)
    sc_bot_path = sc_out_dir / "eve_Bot.json"
    
    s_tot = 0
    s_mat = 0
    s_bot = 0
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
            ts_epoch = parse_ts(ts_str)
            
            if (sip, sport) <= (dip, dport):
                k = (sip, sport, dip, dport, normalize_proto(proto))
            else:
                k = (dip, dport, sip, sport, normalize_proto(proto))
                
            candidates = index.get(k)
            if candidates:
                best_lbl = None
                min_diff = float("inf")
                for (g_ts, g_lbl) in candidates:
                    diff = abs(ts_epoch - g_ts)
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
                            s_bot += 1
                            
    with open(sc_bot_path, "w", encoding="utf-8") as out_f:
        out_f.writelines(sc_lines)
        out_f.flush()
        os.fsync(out_f.fileno())
        
    pct = (s_mat / max(s_tot, 1)) * 100
    print(f"[{sc_name}] Suricata={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Saved={s_bot:,} (Time: {time.time()-t0:.1f}s)")
    return sc_name, s_tot, s_mat, pct, s_bot, sc_lines

def main():
    ctu_dir = Path("data/CTU-13")
    eve_dir = Path("data/eve/CTU-13")
    out_dir = Path("data/relabeled_ctu13")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comb_path = out_dir / "eve_Bot.json"
    all_comb_lines = []
    
    report = []
    start_all = time.time()
    
    for sc_dir in sorted(ctu_dir.glob("scenario_*")):
        sc_name = sc_dir.name
        eve_path = eve_dir / sc_name / "eve.json"
        label_files = list(sc_dir.glob("*.binetflow")) + list(sc_dir.glob("*.biargus"))
        if not eve_path.exists() or not label_files:
            continue
            
        sc_name, s_tot, s_mat, pct, s_bot, sc_lines = process_scenario(
            sc_name, eve_path, label_files[0], out_dir
        )
        report.append((sc_name, s_tot, s_mat, pct, s_bot))
        all_comb_lines.extend(sc_lines)
        
    with open(comb_path, "w", encoding="utf-8") as comb_f:
        comb_f.writelines(all_comb_lines)
        comb_f.flush()
        os.fsync(comb_f.fileno())
        
    dur = time.time() - start_all
    tot_sur = sum(r[1] for r in report)
    tot_mat = sum(r[2] for r in report)
    tot_bot = sum(r[4] for r in report)
    tot_pct = (tot_mat / max(tot_sur, 1)) * 100
    
    print("\n" + "=" * 80)
    print("  CTU-13 RELABELING FINAL REPORT (AŞAMA 1)")
    print("=" * 80)
    print(f"{'Scenario':<15} {'Suricata Flows':>16} {'Matched Flows':>16} {'Match Rate':>12} {'Botnet Flows':>16}")
    print("-" * 80)
    for sc_name, s_tot, s_mat, s_pct, s_bot in report:
        print(f"{sc_name:<15} {s_tot:>16,} {s_mat:>16,} {s_pct:>11.2f}% {s_bot:>16,}")
    print("-" * 80)
    print(f"{'TOTAL / OVERALL':<15} {tot_sur:>16,} {tot_mat:>16,} {tot_pct:>11.2f}% {tot_bot:>16,}")
    print("=" * 80)
    print(f"  Combined Output File : {comb_path} ({len(all_comb_lines):,} lines)")
    print(f"  Total Duration       : {dur/60:.2f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
