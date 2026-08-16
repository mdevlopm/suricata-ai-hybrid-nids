#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

mkdir -p data/eve/CTU-13

scenarios=(
  "scenario_1:data/raw_pcap/Ctu-13/botnet-capture-20110810-neris.pcap"
  "scenario_2:data/raw_pcap/Ctu-13/botnet-capture-20110811-neris.pcap"
  "scenario_3:data/raw_pcap/Ctu-13/botnet-capture-20110812-rbot.pcap"
  "scenario_4:data/raw_pcap/Ctu-13/botnet-capture-20110815-fast-flux.pcap"
  "scenario_5:data/raw_pcap/Ctu-13/botnet-capture-20110815-fast-flux-2.pcap"
  "scenario_6:data/raw_pcap/Ctu-13/botnet-capture-20110815-rbot-dos.pcap"
  "scenario_7:data/raw_pcap/Ctu-13/botnet-capture-20110816-donbot.pcap"
  "scenario_8:data/raw_pcap/Ctu-13/botnet-capture-20110816-qvod.pcap"
  "scenario_9:data/raw_pcap/Ctu-13/botnet-capture-20110816-sogou.pcap"
  "scenario_10:data/raw_pcap/Ctu-13/botnet-capture-20110817-bot.pcap"
  "scenario_11:data/raw_pcap/Ctu-13/botnet-capture-20110818-bot-2.pcap"
  "scenario_13:data/raw_pcap/Ctu-13/botnet-capture-20110819-bot.pcap"
)

for entry in "${scenarios[@]}"; do
  scenario="${entry%%:*}"
  pcap="${entry#*:}"
  out_dir="data/eve/CTU-13/$scenario"
  
  if [ -f "$out_dir/eve.json" ] && [ -s "$out_dir/eve.json" ]; then
    echo "$(date) Skipping $scenario (eve.json already exists: $(ls -lh "$out_dir/eve.json" | awk '{print $5}'))"
    continue
  fi

  if [ ! -f "$pcap" ]; then
    echo "$(date) Warning: PCAP $pcap not found, skipping."
    continue
  fi

  echo "$(date) Processing $scenario ($pcap)..."
  mkdir -p "$out_dir"
  suricata -c config/suricata.yaml -r "$pcap" -l "$out_dir" -k none
  echo "$(date) Completed $scenario: $(ls -lh "$out_dir/eve.json" | awk '{print $5}')"
done

echo "$(date) Suricata CTU-13 batch processing complete."
