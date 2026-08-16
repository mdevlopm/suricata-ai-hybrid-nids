#!/usr/bin/env python3
import json
import time
import os
import sys
from pathlib import Path

# Add pipeline directory to sys.path for extract_features_v7
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from trainv8 import extract_features_v7
from scripts.relabel_ctu13 import ArgusLabelIndex, is_botnet_label
from dateutil.parser import isoparse

def main():
    ctu_dir = Path("data/CTU-13")
    eve_dir = Path("data/eve/CTU-13")
    out_dir = Path("data/relabeled_ctu13")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    comb_path = out_dir / "eve_Bot.json"
    if comb_path.exists():
        comb_path.unlink()
        
    tot_sur = 0
    tot_mat = 0
    tot_bot = 0
    
    report = []
    
    start_all = time.time()
    
    for sc_dir in sorted(ctu_dir.glob("scenario_*")):
        sc_name = sc_dir.name
        eve_path = eve_dir / sc_name / "eve.json"
        label_files = list(sc_dir.glob("*.binetflow")) + list(sc_dir.glob("*.biargus"))
        if not eve_path.exists() or not label_files:
            continue
            
        sc_out_dir = out_dir / sc_name
        sc_out_dir.mkdir(parents=True, exist_ok=True)
        sc_bot_path = sc_out_dir / "eve_Bot.json"
        
        index = ArgusLabelIndex(label_files[0])
        
        s_tot = 0
        s_mat = 0
        s_bot = 0
        
        with open(sc_bot_path, "w", encoding="utf-8") as out_f, \
             open(eve_path, "r", encoding="utf-8", errors="replace") as in_f:
            for line in in_f:
                line = line.strip()
                if not line: continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event_type") != "flow": continue
                s_tot += 1
                
                flow = ev.get("flow", {})
                src_ip = ev.get("src_ip", "")
                src_port = int(ev.get("src_port", 0) or 0)
                dest_ip = ev.get("dest_ip", "")
                dest_port = int(ev.get("dest_port", 0) or 0)
                proto = ev.get("proto", "tcp")
                ts_str = flow.get("start") or ev.get("timestamp", "")
                ts_epoch = isoparse(ts_str).timestamp()
                
                gt_lbl = index.match(src_ip, src_port, dest_ip, dest_port, proto, ts_epoch, window_s=5.0)
                if gt_lbl is not None:
                    s_mat += 1
                    if is_botnet_label(gt_lbl):
                        feat = extract_features_v7(ev)
                        if feat is not None:
                            ev["gt_label"] = gt_lbl
                            ev["ctu_scenario"] = sc_name
                            out_f.write(json.dumps(ev) + "\n")
                            s_bot += 1
            out_f.flush()
            os.fsync(out_f.fileno())
            
        pct = (s_mat / max(s_tot, 1)) * 100
        report.append((sc_name, s_tot, s_mat, pct, s_bot))
        tot_sur += s_tot
        tot_mat += s_mat
        tot_bot += s_bot
        print(f"[{sc_name}] Suricata={s_tot:,} | Matched={s_mat:,} ({pct:.2f}%) | Botnet Saved={s_bot:,}")
        
    # Append to combined eve_Bot.json
    with open(comb_path, "w", encoding="utf-8") as comb_f:
        for sc_name, _, _, _, _ in report:
            sc_bot_path = out_dir / sc_name / "eve_Bot.json"
            if sc_bot_path.exists():
                with open(sc_bot_path, "r", encoding="utf-8") as sf:
                    for l in sf:
                        comb_f.write(l)
        comb_f.flush()
        os.fsync(comb_f.fileno())
        
    dur = time.time() - start_all
    print("\n" + "=" * 80)
    print("  CTU-13 RELABELING FINAL REPORT (AŞAMA 1)")
    print("=" * 80)
    print(f"{'Scenario':<15} {'Suricata Flows':>16} {'Matched Flows':>16} {'Match Rate':>12} {'Botnet Flows':>16}")
    print("-" * 80)
    for sc_name, s_tot, s_mat, s_pct, s_bot in report:
        print(f"{sc_name:<15} {s_tot:>16,} {s_mat:>16,} {s_pct:>11.2f}% {s_bot:>16,}")
    print("-" * 80)
    tot_pct = (tot_mat / max(tot_sur, 1)) * 100
    print(f"{'TOTAL / OVERALL':<15} {tot_sur:>16,} {tot_mat:>16,} {tot_pct:>11.2f}% {tot_bot:>16,}")
    print("=" * 80)
    print(f"  Combined Output File : {comb_path}")
    print(f"  Total Duration       : {dur/60:.2f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
