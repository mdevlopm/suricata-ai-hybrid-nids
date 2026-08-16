#!/usr/bin/env python3
import json
import time
import os
import sys
import subprocess
import glob
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import re

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

def process_single_scenario(sc_dir: Path, eve_sc_dir: Path, config_path: Path, out_dir: Path, comb_file_handle):
    sc_name = sc_dir.name
    malware_family = get_malware_family(sc_dir)
    eve_sc_dir.mkdir(parents=True, exist_ok=True)
    eve_file = eve_sc_dir / "eve.json"
    
    if not eve_file.exists() or eve_file.stat().st_size == 0:
        pcap_files = list(sc_dir.glob("*.pcap")) + list(sc_dir.glob("**/*.pcap"))
        if pcap_files:
            for pcap in pcap_files:
                cmd = ["suricata", "-c", str(config_path), "-r", str(pcap), "-l", str(eve_sc_dir)]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                except Exception as e:
                    print(f"Suricata error processing {pcap.name}: {e}")

    if not eve_file.exists():
        return sc_name, 0, 0, 0.0, 0

    # Check ground truth
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

    sc_out = out_dir / sc_name
    sc_out.mkdir(parents=True, exist_ok=True)
    sc_bot_path = sc_out / "eve_Bot.json"

    s_tot, s_mat, s_bot = 0, 0, 0

    with open(sc_bot_path, "w", encoding="utf-8") as sf, open(eve_file, "r", encoding="utf-8", errors="replace") as in_f:
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
                        sf.write(l_str)
                        if comb_file_handle:
                            comb_file_handle.write(l_str)
                        s_bot += 1

        sf.flush()
        os.fsync(sf.fileno())

    pct = (s_mat / max(s_tot, 1)) * 100
    return sc_name, s_tot, s_mat, pct, s_bot

def main():
    base_dir = Path("/run/media/mehmet/siber data1/ai modeli xgboost")
    mcfp_dir = base_dir / "data/MCFP"
    raw_mcfp_json = base_dir / "data/raw_pcap/mcfp felk/eve_botnet_mcfp.json"
    eve_base_dir = base_dir / "data/eve/MCFP"
    out_dir = base_dir / "data/relabeled_mcfp"
    config_path = base_dir / "config/suricata.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_file = out_dir / ".completed_scenarios.txt"

    scenarios = sorted([d for d in mcfp_dir.glob("CTU-Malware-Capture-Botnet-*") if d.is_dir()])
    
    completed_names = set()
    if ck_file.exists():
        completed_names = set(line.strip() for line in ck_file.read_text().splitlines() if line.strip())

    comb_path = out_dir / "eve_Bot.json"
    start_all = time.time()
    report = []

    with open(comb_path, "a", encoding="utf-8") as cf:
        if scenarios:
            print(f"[MCFP Pipeline] Başladı, {len(scenarios)} scenario klasörü tespit edildi.")
            for sc_dir in scenarios:
                sc_name = sc_dir.name
                if sc_name in completed_names:
                    print(f"[{sc_name}] Zaten tamamlanmış (checkpoint), atlanıyor.")
                    continue

                eve_sc_dir = eve_base_dir / sc_name
                name, s_tot, s_mat, pct, s_bot = process_single_scenario(sc_dir, eve_sc_dir, config_path, out_dir, cf)
                cf.flush()
                os.fsync(cf.fileno())
                
                with open(ck_file, "a", encoding="utf-8") as ck:
                    ck.write(sc_name + "\n")
                    ck.flush()

                report.append((sc_name, s_tot, s_mat, pct, s_bot))
                print(f"[{sc_name}] Suricata={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Saved={s_bot:,}")
        elif raw_mcfp_json.exists():
            sc_name = "mcfp_botnet"
            if sc_name in completed_names:
                print(f"[{sc_name}] Zaten tamamlanmış (checkpoint).")
            else:
                print(f"[MCFP Pipeline] Başladı, 33.5GB toplu EVE logu ({raw_mcfp_json.name}) işleniyor.")
                sc_out = out_dir / sc_name
                sc_out.mkdir(parents=True, exist_ok=True)
                sc_bot_path = sc_out / "eve_Bot.json"

                s_tot, s_mat, s_bot = 0, 0, 0

                with open(sc_bot_path, "w", encoding="utf-8") as sf, open(raw_mcfp_json, "r", encoding="utf-8", errors="replace") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line or '"event_type":"flow"' not in line: continue
                        try: ev = json.loads(line)
                        except Exception: continue
                        if ev.get("event_type") != "flow": continue
                        s_tot += 1
                        s_mat += 1

                        feat = extract_features_v7(ev)
                        if feat is not None:
                            ev["gt_label"] = "Botnet"
                            ev["mcfp_scenario"] = sc_name
                            ev["malware_family"] = "Bot"
                            l_str = json.dumps(ev) + "\n"
                            sf.write(l_str)
                            cf.write(l_str)
                            s_bot += 1

                        if s_tot % 2000000 == 0:
                            sf.flush()
                            cf.flush()
                            print(f"  Processed {s_tot:,} flow events, saved {s_bot:,} Botnet flows...")

                    sf.flush()
                    os.fsync(sf.fileno())

                cf.flush()
                os.fsync(cf.fileno())
                with open(ck_file, "a", encoding="utf-8") as ck:
                    ck.write(sc_name + "\n")
                    ck.flush()

                pct = 100.0
                report.append((sc_name, s_tot, s_mat, pct, s_bot))
                print(f"[mcfp_botnet] Suricata={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Saved={s_bot:,}")

    dur = time.time() - start_all
    tot_sur = sum(r[1] for r in report)
    tot_mat = sum(r[2] for r in report)
    tot_bot = sum(r[4] for r in report)
    tot_pct = (tot_mat / max(tot_sur, 1)) * 100

    print("\n" + "=" * 80)
    print("  MCFP RELABELING FINAL VERIFIED SUMMARY (AŞAMA 2)")
    print("=" * 80)
    print(f"{'Scenario':<30} {'Suricata Flows':>16} {'Matched Flows':>16} {'Match Rate':>12} {'Botnet Flows':>16}")
    print("-" * 80)
    for sc_name, s_tot, s_mat, s_pct, s_bot in report:
        print(f"{sc_name:<30} {s_tot:>16,} {s_mat:>16,} {s_pct:>11.2f}% {s_bot:>16,}")
    print("-" * 80)
    print(f"{'TOTAL / OVERALL':<30} {tot_sur:>16,} {tot_mat:>16,} {tot_pct:>11.2f}% {tot_bot:>16,}")
    print("=" * 80)
    print(f"  Combined Output File : {comb_path}")
    print(f"  Total Duration       : {dur/60:.2f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
