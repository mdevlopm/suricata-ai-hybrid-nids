#!/usr/bin/env python3
"""
CTU-13 Dataset Relabeling & Feature Extraction Runner
------------------------------------------------------
Fully self-contained batch relabeling and feature extraction script for CTU-13.
Guarantees fsync persistence across external drives.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from dateutil.parser import isoparse

# Add pipeline directory to import extract_features_v7
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7
from relabel_ctu13 import ArgusLabelIndex, is_botnet_label


def parse_timestamp_epoch(ts_str: str) -> float:
    """Parses ISO or Argus timestamp string to epoch seconds."""
    if not ts_str:
        return 0.0
    ts_str = str(ts_str).strip()
    try:
        if "T" in ts_str:
            return isoparse(ts_str).timestamp()
            
        if (len(ts_str) >= 19) and (ts_str[4] in ("/", "-")) and (ts_str[7] in ("/", "-")):
            dt = datetime(
                int(ts_str[0:4]), int(ts_str[5:7]), int(ts_str[8:10]),
                int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
                tzinfo=timezone.utc
            )
            micros = 0.0
            dot_idx = ts_str.find(".")
            if dot_idx != -1:
                m_digits = "".join([c for c in ts_str[dot_idx+1:dot_idx+7] if c.isdigit()])
                if m_digits:
                    micros = float(f"0.{m_digits}")
            return dt.timestamp() + micros
    except Exception:
        pass
    return 0.0


def main():
    ctu_dir = Path("data/CTU-13")
    eve_dir = Path("data/eve/CTU-13")
    out_dir = Path("data/relabeled_ctu13")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_bot_path = out_dir / "eve_Bot.json"
    with open(combined_bot_path, "w", encoding="utf-8") as f:
        f.flush()
        os.fsync(f.fileno())

    scenarios = sorted([d for d in eve_dir.glob("scenario_*") if d.is_dir()], key=lambda x: int(x.name.split("_")[1]))

    print("=" * 80)
    print("  CTU-13 BATCH RELABELING & FEATURE EXTRACTION RUNNER")
    print("=" * 80)

    total_suricata = 0
    total_matched = 0
    total_botnet = 0
    scenario_report = []

    start_total_t = time.time()

    for sc_dir in scenarios:
        sc_name = sc_dir.name
        sc_ctu_dir = ctu_dir / sc_name
        eve_json = sc_dir / "eve.json"
        
        label_files = list(sc_ctu_dir.glob("*.binetflow")) + list(sc_ctu_dir.glob("*.biargus"))
        if not label_files:
            continue
            
        label_file = label_files[0]
        sc_out_dir = out_dir / sc_name
        sc_out_dir.mkdir(parents=True, exist_ok=True)
        sc_bot_file = sc_out_dir / "eve_Bot.json"

        print(f"\nProcessing {sc_name}...")
        index = ArgusLabelIndex(label_file)

        s_tot = 0
        s_mat = 0
        s_bot = 0

        with open(sc_bot_file, "w", encoding="utf-8") as out_f, open(eve_json, "r", encoding="utf-8", errors="replace") as in_f:
            for line in in_f:
                line = line.strip()
                if not line: continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event_type") != "flow":
                    continue

                s_tot += 1
                flow_data = event.get("flow", {})
                src_ip = event.get("src_ip", "")
                src_port = int(event.get("src_port", 0) or 0)
                dest_ip = event.get("dest_ip", "")
                dest_port = int(event.get("dest_port", 0) or 0)
                proto = event.get("proto", "tcp")

                ts_str = flow_data.get("start") or event.get("timestamp", "")
                ts_epoch = parse_timestamp_epoch(ts_str)

                gt_label = index.match(src_ip, src_port, dest_ip, dest_port, proto, ts_epoch, window_s=5.0)
                if gt_label is not None:
                    s_mat += 1
                    if is_botnet_label(gt_label):
                        feat = extract_features_v7(event)
                        if feat is not None:
                            event["gt_label"] = gt_label
                            event["ctu_scenario"] = sc_name
                            out_f.write(json.dumps(event) + "\n")
                            s_bot += 1

            out_f.flush()
            os.fsync(out_f.fileno())

        print(f"  --> [{sc_name}] Wrote {s_bot:,} Botnet flow lines to {sc_bot_file}")
        
        # Append scenario bot flows to combined eve_Bot.json
        if sc_bot_file.exists():
            with open(sc_bot_file, "r", encoding="utf-8") as sc_f, \
                 open(combined_bot_path, "a", encoding="utf-8") as comb_f:
                for line in sc_f:
                    comb_f.write(line)
                comb_f.flush()
                os.fsync(comb_f.fileno())

        total_suricata += s_tot
        total_matched += s_mat
        total_botnet += s_bot
        match_pct = (s_mat / max(s_tot, 1)) * 100.0
        scenario_report.append((sc_name, s_tot, s_mat, match_pct, s_bot))

    total_elapsed = time.time() - start_total_t
    overall_match_pct = (total_matched / max(total_suricata, 1)) * 100.0

    print("\n" + "=" * 80)
    print("  CTU-13 RELABELING FINAL REPORT (AŞAMA 1)")
    print("=" * 80)
    print(f"{'Scenario':<15} {'Suricata Flows':>16} {'Matched Flows':>16} {'Match Rate':>12} {'Botnet Flows':>16}")
    print("-" * 80)
    for sc_name, s_tot, s_mat, s_pct, s_bot in scenario_report:
        print(f"{sc_name:<15} {s_tot:>16,} {s_mat:>16,} {s_pct:>11.2f}% {s_bot:>16,}")
    print("-" * 80)
    print(f"{'TOTAL / OVERALL':<15} {total_suricata:>16,} {total_matched:>16,} {overall_match_pct:>11.2f}% {total_botnet:>16,}")
    print("=" * 80)
    print(f"  Combined Output File : {combined_bot_path}")
    print(f"  Total Duration       : {total_elapsed/60.0:.2f} min ({total_suricata/max(total_elapsed, 0.001):,.0f} flow/s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
