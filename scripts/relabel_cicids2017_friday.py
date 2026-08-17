#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth Relabeling for CICIDS2017 Friday (Friday-WorkingHours.pcap)
Properly attributes both external attacker (205.174.165.73), attack subnet router (172.16.0.1),
victim servers (192.168.10.50, 192.168.10.51), and infected bot hosts.
Extracts:
- Botnet (ARES)
- DDoS (LOIC)
- PortScan
- Benign (pure background office traffic)
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from dateutil.parser import isoparse

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7

ATTACKER_IPS = {"205.174.165.73", "172.16.0.1"}
BOT_VICTIMS = {"192.168.10.15", "192.168.10.9", "192.168.10.14", "192.168.10.5", "192.168.10.8"}
SERVER_VICTIMS = {"192.168.10.50", "192.168.10.51"}

def parse_utc_dt(ts_str: str) -> datetime:
    if not ts_str: return None
    try:
        dt = isoparse(str(ts_str).strip())
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def main():
    base_dir = Path(__file__).resolve().parent.parent
    eve_path = base_dir / "data/eve/cicids2017_friday/eve.json"
    out_dir = base_dir / "data/relabeled_cicids2017"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eve_path.exists():
        print(f"Error: {eve_path} does not exist yet!")
        return

    print("=" * 80)
    print("RELABELING CICIDS2017 FRIDAY EVE FLOWS (CORRECTED ATTACK ATTRIBUTION)")
    print(f"Source: {eve_path}")
    print("=" * 80)

    bot_lines = []
    ddos_lines = []
    portscan_lines = []
    benign_lines = []

    total_flows = 0
    start_time = time.time()

    with open(eve_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip(): continue
            try: ev = json.loads(line)
            except Exception: continue
            if ev.get("event_type") != "flow": continue
            total_flows += 1

            sip = ev.get("src_ip", "")
            dip = ev.get("dest_ip", "")
            flow = ev.get("flow", {})
            st_str = flow.get("start") or ev.get("timestamp", "")
            dt_utc = parse_utc_dt(st_str)
            if dt_utc is None: continue

            h = dt_utc.hour
            m = dt_utc.minute
            time_min = h * 60 + m

            label = "Benign"

            # 1. LOIC DDoS Attack: 172.16.0.1 or 205.174.165.73 -> 192.168.10.50 (18:50 - 19:30 UTC or general flood to .50)
            if (sip in ATTACKER_IPS and dip in SERVER_VICTIMS) or (dip in ATTACKER_IPS and sip in SERVER_VICTIMS):
                label = "DDoS"
            elif sip == "172.16.0.1" and dip == "192.168.10.50":
                label = "DDoS"
            elif dip == "192.168.10.50" and ev.get("proto") == "TCP" and 1130 <= time_min <= 1180:
                label = "DDoS"
            # 2. PortScan Attack: 16:50 to 18:40 UTC (1010 to 1120 mins) from attacker IPs
            elif (sip in ATTACKER_IPS or dip in ATTACKER_IPS) and 1010 <= time_min <= 1120:
                label = "PortScan"
            # 3. Botnet (ARES): 13:00 to 14:15 UTC (780 to 855 mins)
            elif (sip in ATTACKER_IPS or dip in ATTACKER_IPS) and 780 <= time_min <= 855:
                label = "Bot"
            elif (sip in BOT_VICTIMS and dip in ATTACKER_IPS) or (dip in BOT_VICTIMS and sip in ATTACKER_IPS):
                label = "Bot"
            # 4. Any remaining traffic directly with external attacker IP
            elif sip == "205.174.165.73" or dip == "205.174.165.73":
                label = "PortScan" if 1010 <= time_min <= 1120 else "DDoS"
            else:
                label = "Benign"

            fv = extract_features_v7(ev)
            if fv is not None:
                ev["gt_label"] = label
                ev["source_dataset"] = "CICIDS2017_Friday"
                ev_str = json.dumps(ev) + "\n"

                if label == "Bot":
                    bot_lines.append(ev_str)
                elif label == "DDoS":
                    ddos_lines.append(ev_str)
                elif label == "PortScan":
                    portscan_lines.append(ev_str)
                elif label == "Benign":
                    benign_lines.append(ev_str)

            if total_flows % 50_000 == 0:
                print(f"  Processed {total_flows:,} flows... (Bot={len(bot_lines):,}, DDoS={len(ddos_lines):,}, PortScan={len(portscan_lines):,}, Benign={len(benign_lines):,})")

    # Write output files
    with open(out_dir / "eve_Bot.json", "w", encoding="utf-8") as f:
        f.writelines(bot_lines)
    with open(out_dir / "eve_DDoS.json", "w", encoding="utf-8") as f:
        f.writelines(ddos_lines)
    with open(out_dir / "eve_PortScan.json", "w", encoding="utf-8") as f:
        f.writelines(portscan_lines)
    with open(out_dir / "eve_Benign.json", "w", encoding="utf-8") as f:
        f.writelines(benign_lines)

    print("\n" + "=" * 80)
    print(f"RELABELING COMPLETE in {time.time() - start_time:.1f}s")
    print(f"Total Suricata Flows Processed : {total_flows:,}")
    print(f"  • Botnet Flows Saved         : {len(bot_lines):,}")
    print(f"  • DDoS Flows Saved           : {len(ddos_lines):,}")
    print(f"  • PortScan Flows Saved       : {len(portscan_lines):,}")
    print(f"  • Truly Clean Benign Flows   : {len(benign_lines):,}")
    print(f"Outputs written to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
