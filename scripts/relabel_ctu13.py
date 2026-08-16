#!/usr/bin/env python3
"""
CTU-13 Dataset Relabeling & Feature Extraction Script (Aşama 1)
----------------------------------------------------------------
1. Match Suricata eve.json flows with Argus biargus / binetflow ground-truth labels.
2. Matching criteria: (src_ip, sport, dst_ip, dport, proto) [bidirectional] + timestamp (±5s window).
3. Ground-truth labels come ONLY from biargus/binetflow (Background, Normal, Botnet, C&C).
4. Extract 70 features using extract_features_v7() ONLY (NO tshark enrichment).
5. Output ONLY Botnet/C&C flows to data/relabeled_ctu13/$scenario/eve_Bot.json and combined data/relabeled_ctu13/eve_Bot.json.
6. True line-by-line streaming & low memory footprint.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Insert pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7


from datetime import timezone
from dateutil.parser import isoparse


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


def normalize_proto(proto_str: str) -> str:
    """Normalizes protocol string to lowercase standard."""
    p = str(proto_str).lower().strip()
    if p in ("6", "tcp"): return "tcp"
    if p in ("17", "udp"): return "udp"
    if p in ("1", "icmp"): return "icmp"
    return p


class ArgusLabelIndex:
    """Indexes Argus binetflow/biargus entries for fast 5-tuple + time window lookup."""
    
    def __init__(self, label_path: Path):
        self.label_path = label_path
        # Index: (ip1, port1, ip2, port2, proto) -> list of (epoch, duration, label_str)
        self.index = defaultdict(list)
        self.total_entries = 0
        self._load()

    def _load(self):
        if not self.label_path.exists():
            print(f"  [WARN] Label file not found: {self.label_path}")
            return
        
        print(f"  Indexing ground-truth labels from {self.label_path.name}...")
        start_t = time.time()
        
        with open(self.label_path, "r", encoding="utf-8", errors="replace") as f:
            header_read = False
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split(",")
                if not header_read and "StartTime" in parts[0]:
                    header_read = True
                    continue
                
                if len(parts) < 15:
                    continue
                
                # StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label
                ts_str = parts[0]
                proto  = normalize_proto(parts[2])
                saddr  = parts[3].strip()
                sport  = int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                daddr  = parts[6].strip()
                dport  = int(parts[7].strip()) if parts[7].strip().isdigit() else 0
                label  = parts[14].strip()
                
                ts_epoch = parse_timestamp_epoch(ts_str)
                if ts_epoch <= 0.0:
                    continue
                
                # Canonical 5-tuple key (sorted endpoint pairs for bidirectional matching)
                if (saddr, sport) <= (daddr, dport):
                    key = (saddr, sport, daddr, dport, proto)
                else:
                    key = (daddr, dport, saddr, sport, proto)
                
                self.index[key].append((ts_epoch, label))
                self.total_entries += 1
                
        elapsed = time.time() - start_t
        print(f"  Indexed {self.total_entries:,} ground-truth flows in {elapsed:.2f}s ({len(self.index):,} unique 5-tuples)")

    def match(self, saddr: str, sport: int, daddr: str, dport: int, proto: str, ts_epoch: float, window_s: float = 5.0) -> str:
        """Finds matching ground-truth label within time window (±window_s)."""
        proto = normalize_proto(proto)
        if (saddr, sport) <= (daddr, dport):
            key = (saddr, sport, daddr, dport, proto)
        else:
            key = (daddr, dport, saddr, sport, proto)
        
        candidates = self.index.get(key)
        if not candidates:
            return None
        
        best_label = None
        min_diff = window_s + 1.0
        
        for cand_ts, cand_label in candidates:
            diff = abs(ts_epoch - cand_ts)
            if diff <= window_s and diff < min_diff:
                min_diff = diff
                best_label = cand_label
                
        return best_label


def is_botnet_label(label_str: str) -> bool:
    """Checks if ground-truth label represents Botnet / C&C traffic."""
    if not label_str:
        return False
    l_lower = label_str.lower()
    return "botnet" in l_lower or "c&c" in l_lower or "cc" in l_lower or "bot" in l_lower


def process_scenario(scenario_name: str, eve_path: Path, label_path: Path, out_dir: Path, time_window_s: float = 5.0):
    """Processes a single scenario's eve.json against its biargus/binetflow ground-truth labels."""
    print(f"\n============================================================")
    print(f"  PROCESSING SCENARIO: {scenario_name}")
    print(f"============================================================")
    print(f"  EVE Path   : {eve_path}")
    print(f"  Label Path : {label_path}")
    print(f"  Out Dir    : {out_dir}")
    
    if not eve_path.exists():
        print(f"  [ERROR] eve.json not found: {eve_path}")
        return 0, 0, 0, 0
    
    index = ArgusLabelIndex(label_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bot_path = out_dir / "eve_Bot.json"
    
    stats = {
        "suricata_flows": 0,
        "matched_flows": 0,
        "botnet_flows": 0,
        "background_flows": 0,
        "normal_flows": 0,
        "unmatched_flows": 0
    }
    
    start_time = time.time()
    
    written_botnet = 0
    with open(out_bot_path, "w", encoding="utf-8") as out_f, \
         open(eve_path, "r", encoding="utf-8", errors="replace") as in_f:
        
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
            except Exception:
                continue
            
            if event.get("event_type") != "flow":
                continue
            
            stats["suricata_flows"] += 1
            
            # Extract flow 5-tuple and timestamp
            flow_data = event.get("flow", {})
            src_ip = event.get("src_ip", "")
            src_port = int(event.get("src_port", 0) or 0)
            dest_ip = event.get("dest_ip", "")
            dest_port = int(event.get("dest_port", 0) or 0)
            proto = event.get("proto", "tcp")
            
            ts_str = flow_data.get("start") or event.get("timestamp", "")
            ts_epoch = parse_timestamp_epoch(ts_str)
            
            # Match against ground-truth index
            gt_label = index.match(src_ip, src_port, dest_ip, dest_port, proto, ts_epoch, window_s=time_window_s)
            
            if gt_label is not None:
                stats["matched_flows"] += 1
                if is_botnet_label(gt_label):
                    stats["botnet_flows"] += 1
                    
                    # Feature Extraction (extract_features_v7 ONLY)
                    feat = extract_features_v7(event)
                    if feat is not None:
                        # Append ground-truth label metadata to event
                        event["gt_label"] = gt_label
                        event["ctu_scenario"] = scenario_name
                        out_f.write(json.dumps(event) + "\n")
                        written_botnet += 1
                elif "normal" in gt_label.lower():
                    stats["normal_flows"] += 1
                else:
                    stats["background_flows"] += 1
            else:
                stats["unmatched_flows"] += 1
                
            if stats["suricata_flows"] % 100_000 == 0:
                elapsed = time.time() - start_time
                print(f"  ... {stats['suricata_flows']:,} flows processed | "
                      f"Matched: {stats['matched_flows']:,} | "
                      f"Botnet: {stats['botnet_flows']:,} ({stats['suricata_flows']/max(elapsed,0.001):,.0f} flow/s)")
                      
        out_f.flush()
        os.fsync(out_f.fileno())
        
    print(f"  --> Wrote {written_botnet:,} Botnet flow lines to {out_bot_path}")
                
    elapsed = time.time() - start_time
    match_rate = (stats["matched_flows"] / max(stats["suricata_flows"], 1)) * 100.0
    bot_rate = (stats["botnet_flows"] / max(stats["suricata_flows"], 1)) * 100.0
    
    print(f"\n  SCENARIO {scenario_name} SUMMARY:")
    print(f"  Total Suricata Flows : {stats['suricata_flows']:>12,}")
    print(f"  Matched Flows        : {stats['matched_flows']:>12,} ({match_rate:.2f}%)")
    print(f"  └─ Botnet/C&C Flows  : {stats['botnet_flows']:>12,} ({bot_rate:.2f}%) -> Saved to {out_bot_path.name}")
    print(f"  └─ Background Flows  : {stats['background_flows']:>12,}")
    print(f"  └─ Normal Flows      : {stats['normal_flows']:>12,}")
    print(f"  Unmatched Flows      : {stats['unmatched_flows']:>12,}")
    print(f"  Processing Duration  : {elapsed:.2f}s ({stats['suricata_flows']/max(elapsed,0.001):,.0f} flow/s)")
    
    return stats["suricata_flows"], stats["matched_flows"], stats["botnet_flows"], stats["unmatched_flows"]


def main():
    parser = argparse.ArgumentParser(description="CTU-13 Relabeling & Feature Extraction (Aşama 1)")
    parser.add_argument("--ctu_dir", default="data/CTU-13", help="CTU-13 dataset directory")
    parser.add_argument("--eve_dir", default="data/eve/CTU-13", help="Suricata eve.json outputs directory")
    parser.add_argument("--out_dir", default="data/relabeled_ctu13", help="Relabeled output directory")
    parser.add_argument("--window", type=float, default=5.0, help="Timestamp matching window in seconds (default: 5.0)")
    args = parser.parse_args()
    
    ctu_dir = Path(args.ctu_dir)
    eve_dir = Path(args.eve_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = sorted([d for d in eve_dir.glob("scenario_*") if d.is_dir()], key=lambda x: int(x.name.split("_")[1]))
    
    if not scenarios:
        print(f"No scenario directories found under {eve_dir}")
        sys.exit(1)
        
    print(f"\n============================================================")
    print(f"  CTU-13 BATCH RELABELING & FEATURE EXTRACTION (Aşama 1)")
    print(f"============================================================")
    print(f"  Found {len(scenarios)} scenario(s) to process")
    
    scenario_report = []
    total_suricata = 0
    total_matched = 0
    total_botnet = 0
    total_unmatched = 0
    
    # Combined output file for all CTU-13 Botnet flows
    combined_bot_path = out_dir / "eve_Bot.json"
    with open(combined_bot_path, "w", encoding="utf-8") as combined_f:
        pass  # Initialize empty combined file
        
    start_total_t = time.time()
    
    for sc_eve_dir in scenarios:
        sc_name = sc_eve_dir.name
        sc_ctu_dir = ctu_dir / sc_name
        
        eve_json = sc_eve_dir / "eve.json"
        
        # Find matching .binetflow or .biargus label file
        label_files = list(sc_ctu_dir.glob("*.binetflow")) + list(sc_ctu_dir.glob("*.biargus"))
        if not label_files:
            print(f"[SKIP] No label file (.binetflow / .biargus) found for {sc_name} in {sc_ctu_dir}")
            continue
            
        label_file = label_files[0]
        sc_out_dir = out_dir / sc_name
        
        s_total, s_match, s_bot, s_unmatch = process_scenario(
            scenario_name=sc_name,
            eve_path=eve_json,
            label_path=label_file,
            out_dir=sc_out_dir,
            time_window_s=args.window
        )
        
        total_suricata += s_total
        total_matched += s_match
        total_botnet += s_bot
        total_unmatched += s_unmatch
        
        match_pct = (s_match / max(s_total, 1)) * 100.0
        scenario_report.append((sc_name, s_total, s_match, match_pct, s_bot))
        
        # Append scenario bot flows to combined eve_Bot.json
        sc_bot_path = sc_out_dir / "eve_Bot.json"
        if sc_bot_path.exists():
            with open(sc_bot_path, "r", encoding="utf-8") as sc_f, \
                 open(combined_bot_path, "a", encoding="utf-8") as comb_f:
                for line in sc_f:
                    comb_f.write(line)
                comb_f.flush()
                os.fsync(comb_f.fileno())
                    
    total_elapsed = time.time() - start_total_t
    overall_match_pct = (total_matched / max(total_suricata, 1)) * 100.0
    overall_bot_pct = (total_botnet / max(total_suricata, 1)) * 100.0
    
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
    print(f"  Total Duration       : {total_elapsed/60:.2f} min ({total_suricata/max(total_elapsed,0.001):,.0f} flow/s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
