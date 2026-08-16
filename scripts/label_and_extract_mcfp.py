#!/usr/bin/env python3
import json
import time
import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7

def parse_pcap_ts(ts_str: str) -> float:
    if not ts_str: return 0.0
    ts_str = str(ts_str).strip()
    try:
        clean = ts_str.split('+')[0].split('Z')[0][:19].replace('/', '-').replace(' ', 'T')
        dt = datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception: pass
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
    return 'botnet' in l or 'c&c' in l or 'cc' in l or 'bot' in l or 'malware' in l

def make_key(ip1, port1, ip2, port2, proto):
    p1 = (str(ip1).strip(), int(port1 or 0))
    p2 = (str(ip2).strip(), int(port2 or 0))
    return (p1, p2, norm_p(proto)) if p1 <= p2 else (p2, p1, norm_p(proto))

def get_malware_family(sc_dir: Path) -> str:
    readme_files = list(sc_dir.glob("*.html")) + list(sc_dir.glob("*.txt")) + list(sc_dir.glob("README*"))
    for rf in readme_files:
        try:
            content = rf.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'(?:malware|family|botnet|name)[:\s]+([A-Za-z0-9_\-\.]+)', content, re.IGNORECASE)
            if m: return m.group(1)
        except Exception: pass
    return "Bot"

def main():
    parser = argparse.ArgumentParser(description="Process and feature extract a single MCFP scenario.")
    parser.add_argument("--scenario", required=True, help="Path to scenario directory")
    parser.add_argument("--eve", required=True, help="Path to EVE directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    sc_dir = Path(args.scenario)
    eve_dir = Path(args.eve)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sc_name = sc_dir.name
    malware_family = get_malware_family(sc_dir)
    eve_file = eve_dir / "eve.json"

    if not eve_file.exists():
        print(f"[{sc_name}] ERROR: {eve_file} does not exist.")
        sys.exit(1)

    biargus_files = list(sc_dir.glob("*.biargus")) + list(sc_dir.glob("**/*.biargus"))
    binet_files = list(sc_dir.glob("*.binetflow")) + list(sc_dir.glob("**/*.binetflow"))
    gt_file = biargus_files[0] if biargus_files else (binet_files[0] if binet_files else None)

    idx = defaultdict(list)
    if gt_file and gt_file.exists():
        csv_file = sc_dir / "scenario_flows.csv"
        if biargus_files and (not csv_file.exists() or csv_file.stat().st_size == 0):
            try:
                cmd = f"ra -r \"{gt_file}\" -c , > \"{csv_file}\""
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                gt_file = csv_file
            except Exception: pass

        with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("StartTime") or line.startswith("#") or not line.strip(): continue
                parts = line.strip().split(",")
                if len(parts) < 15: continue
                ts_ep = parse_pcap_ts(parts[0])
                if ts_ep <= 0: continue
                proto = parts[2].strip()
                saddr = parts[3].strip()
                sport = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                daddr = parts[6].strip()
                dport = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
                lbl = parts[14].strip()
                k = make_key(saddr, sport, daddr, dport, proto)
                idx[k].append((ts_ep, lbl))

    sc_bot_path = out_dir / "eve_Bot.json"
    s_tot, s_mat, s_bot = 0, 0, 0
    sc_lines = []

    with open(eve_file, "r", encoding="utf-8", errors="replace") as in_f:
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
            ts_ep = parse_pcap_ts(ts_str)

            best_lbl = None
            if idx:
                k = make_key(sip, sport, dip, dport, proto)
                cands = idx.get(k)
                if cands:
                    min_diff = float("inf")
                    for (g_ts, g_lbl) in cands:
                        diff = abs(ts_ep - g_ts)
                        if diff <= 5.0 and diff < min_diff:
                            min_diff = diff
                            best_lbl = g_lbl
            else:
                best_lbl = "Botnet"

            if best_lbl is not None:
                s_mat += 1
                if is_bot(best_lbl):
                    feat = extract_features_v7(ev)
                    if feat is not None:
                        ev["gt_label"] = best_lbl
                        ev["mcfp_scenario"] = sc_name
                        ev["malware_family"] = malware_family
                        l_str = json.dumps(ev) + "\n"
                        sc_lines.append(l_str)
                        s_bot += 1

    with open(sc_bot_path, "w", encoding="utf-8") as sf:
        sf.writelines(sc_lines)
        sf.flush()
        os.fsync(sf.fileno())

    pct = (s_mat / max(s_tot, 1)) * 100
    print(f"[{sc_name}] Suricata={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Saved={s_bot:,}")

if __name__ == "__main__":
    main()
