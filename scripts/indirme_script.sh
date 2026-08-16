#!/usr/bin/env bash
set -e

mkdir -p "data/csv"

S3_BASE="s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms"
REGION="ca-central-1"

FILES=(
  "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
)

echo "=== CICIDS2018 CSV Download Started: $(date) ==="

for file in "${FILES[@]}"; do
  echo "[$(date)] Downloading $file..."
  aws s3 cp --no-sign-request --region "$REGION" "$S3_BASE/$file" "data/csv/$file"
  echo "[$(date)] Finished $file"
done

echo "=== All Downloads Completed: $(date) ==="
