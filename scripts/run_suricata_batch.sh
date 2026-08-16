#!/bin/bash
# Suricata toplu EVE JSON uretimi — yeni config (flow+tcpflags, http, tls, dns)
set -e

BASE="/run/media/mehmet/siber data1/ai modeli xgboost/data/raw_pcap"
CONFIG="/run/media/mehmet/siber data1/ai modeli xgboost/config/suricata_feature_extract.yaml"
LOG="/tmp/suricata_batch.log"
rm -f "$LOG"

echo "$(date) Basliyor..." | tee -a "$LOG"

cic_dirs=(
  "Wednesday-14-02-2018:Benign:pcap"          # 46G, sadece pcap var
  "Thursday-15-02-2018:DoS:10gb_pcap"        # 11G
  "Friday-16-02-2018:DoS:10gb_pcap"          # 11G
  "Tuesday-20-02-2018:DDoS:10gb_pcap"        # 9.9G
  "Wednesday-21-02-2018:DDoS:10gb_pcap"      # 9.9G
  "Thursday-22-02-2018:WebAttack:10gb_pcap"  # 9.9G
  "Friday-23-02-2018:WebAttack:10gb_pcap"    # 9.9G
  "Wednesday-28-02-2018:Infiltration:10gb_pcap" # 10G
  "Thursday-01-03-2018:Infiltration:10gb_pcap" # 10G
  "Friday-02-03-2018:Bot:10gb_pcap"          # 9.9G
)

# 1) CICIDS2018
for entry in "${cic_dirs[@]}"; do
  IFS=':' read -r dir label pcap_subdir <<< "$entry"
  pcap_path="$BASE/$dir/$pcap_subdir"
  out_path="$BASE/$dir/eve_${label}.json"

  if [ -f "$out_path" ]; then
    echo "$(date) ATLA (zaten var): $out_path" | tee -a "$LOG"
    continue
  fi

  echo "$(date) BASLA: $dir -> $label (pcap: $pcap_path)" | tee -a "$LOG"
  timeout 7200 suricata -c "$CONFIG" -r "$pcap_path" -l "$BASE/$dir" -k none 2>&1 | tail -5 | tee -a "$LOG"

  # suricata eve.json'u her zaman eve.json olarak yazar, biz rename edelim
  if [ -f "$BASE/$dir/eve.json" ]; then
    mv "$BASE/$dir/eve.json" "$out_path"
    echo "$(date) BITTI: $out_path ($(ls -lh "$out_path" | awk '{print $5}'))" | tee -a "$LOG"
  else
    echo "$(date) HATA: eve.json olusmadi $dir" | tee -a "$LOG"
  fi
done

# 2) CTU-13 (sadece neris)
echo "" | tee -a "$LOG"
echo "$(date) CTU-13 BASLIYOR..." | tee -a "$LOG"

ctu_pcaps=(
  "botnet-capture-20110810-neris.pcap"
  "botnet-capture-20110811-neris.pcap"
)
ctu_out="$BASE/Ctu-13/eve_botnet_neris.json"
if [ ! -f "$ctu_out" ]; then
  rm -f "$BASE/Ctu-13/eve.json"
  for pcap in "${ctu_pcaps[@]}"; do
    echo "$(date)  CTU: $pcap" | tee -a "$LOG"
    suricata -c "$CONFIG" -r "$BASE/Ctu-13/$pcap" -l "$BASE/Ctu-13" -k none 2>&1 | tail -2 | tee -a "$LOG"
  done
  if [ -f "$BASE/Ctu-13/eve.json" ]; then
    mv "$BASE/Ctu-13/eve.json" "$ctu_out"
    echo "$(date) CTU BITTI: $ctu_out ($(ls -lh "$ctu_out" | awk '{print $5}'))" | tee -a "$LOG"
  fi
else
  echo "$(date) ATLA (zaten var): $ctu_out" | tee -a "$LOG"
fi

# 3) MCFP
echo "" | tee -a "$LOG"
echo "$(date) MCFP BASLIYOR..." | tee -a "$LOG"
mcfp_out="$BASE/mcfp felk/eve_botnet_mcfp.json"
if [ ! -f "$mcfp_out" ]; then
  rm -f "$BASE/mcfp felk/eve.json"
  # Butun PCAP'lari sirali isle
  find "$BASE/mcfp felk/mcfp_botnet" -name "*.pcap" -print0 | sort -z | while IFS= read -r -d '' pcap; do
    echo "$(date)  MCFP: $(basename "$pcap")" | tee -a "$LOG"
    suricata -c "$CONFIG" -r "$pcap" -l "$BASE/mcfp felk" -k none 2>&1 | tail -1 | tee -a "$LOG"
  done
  if [ -f "$BASE/mcfp felk/eve.json" ]; then
    mv "$BASE/mcfp felk/eve.json" "$mcfp_out"
    echo "$(date) MCFP BITTI: $mcfp_out ($(ls -lh "$mcfp_out" | awk '{print $5}'))" | tee -a "$LOG"
  fi
else
  echo "$(date) ATLA (zaten var): $mcfp_out" | tee -a "$LOG"
fi

# 4) MAWI
echo "" | tee -a "$LOG"
echo "$(date) MAWI BASLIYOR..." | tee -a "$LOG"
mawi_dir="$BASE/Omurga verisi wide"
for pcap in "$mawi_dir"/*.pcap; do
  base=$(basename "$pcap" .pcap)
  out_json="$mawi_dir/suricata_${base}.json"
  if [ -f "$out_json" ]; then
    echo "$(date) ATLA (zaten var): $out_json" | tee -a "$LOG"
    continue
  fi
  echo "$(date) MAWI: $base" | tee -a "$LOG"
  rm -f "$mawi_dir/eve.json"
  timeout 3600 suricata -c "$CONFIG" -r "$pcap" -l "$mawi_dir" -k none 2>&1 | tail -2 | tee -a "$LOG"
  if [ -f "$mawi_dir/eve.json" ]; then
    mv "$mawi_dir/eve.json" "$out_json"
    echo "$(date) BITTI: $out_json ($(ls -lh "$out_json" | awk '{print $5}'))" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "$(date) TUM ISLEMLER TAMAMLANDI" | tee -a "$LOG"
echo "=== OZET ===" | tee -a "$LOG"
find "$BASE" -name "eve_*.json" -exec ls -lh {} \; 2>/dev/null | tee -a "$LOG"
find "$BASE" -name "suricata_2*.json" -exec ls -lh {} \; 2>/dev/null | tee -a "$LOG"
