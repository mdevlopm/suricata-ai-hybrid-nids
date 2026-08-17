# Suricata AI Hybrid Network Intrusion Detection System (NIDS)
### Multi-Source Supervised XGBoost + Temporal BiLSTM Deep Inspection Engine

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Suricata Compatibility](https://img.shields.io/badge/Suricata-EVE_JSON_Native-orange.svg)](https://suricata.io)
[![Production Status](https://img.shields.io/badge/Production_Model-v10c_Baseline-green.svg)](models/baseline/)
[![False Alarm Rate](https://img.shields.io/badge/FAR-0.42%25%20%40%20T%3D0.84-brightgreen.svg)](docs/THRESHOLD_AND_OPERATING_CURVES.md)
[![Attack Recall](https://img.shields.io/badge/Recall-83.41%25%20%40%20T%3D0.84-success.svg)](docs/THRESHOLD_AND_OPERATING_CURVES.md)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## 1. Executive Summary & Project Objective

Traditional signature-based Network Intrusion Detection Systems (NIDS) like Suricata excel at identifying known exploit patterns, CVEs, and static byte sequences. However, modern corporate networks face significant challenges from zero-day vulnerabilities, encrypted evasion tactics, multi-stage advanced persistent threats (APTs), and distributed denial-of-service (DDoS) campaigns.

This project delivers a **two-stage, enterprise-grade AI-powered Hybrid NIDS** operating directly on native Suricata `eve.json` telemetry. The system addresses the fundamental trade-off in network security machine learning: **maintaining an ultra-low False Alarm Rate (FAR $\le 0.50\%$) on benign enterprise traffic while achieving high cross-dataset detection recall ($\ge 80\%$) across all major attack vectors**.

```
                           SURICATA REAL-TIME TELEMETRY
                                (eve.json streaming)
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   Single-Pass Feature Extraction       │
                     │   70 Flow & Protocol-Aware Features    │
                     └───────────────────┬────────────────────┘
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │   STAGE 1: Multi-Source XGBoost        │
                     │   High-Throughput Fast-Path (T = 0.84) │
                     └───────┬────────────────────────┬───────┘
                             │                        │
       [P(Attack) < 0.84]    │                        │  [P(Attack) >= 0.84]
              ▼              ▼                        ▼
        ┌──────────┐   ┌───────────────┐        ┌──────────────────┐
        │  BENIGN  │   │   DoS / DDoS  │        │ Slow / Multi-Hop │
        │ (Ignored)│   │ Fast-Path Alert│       │  Attack Classes  │
        └──────────┘   └───────────────┘        └────────┬─────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │ IP Sliding Window│
                                                │ (L = 40 flows)   │
                                                │ + 8 Behavioral   │
                                                └────────┬─────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │  STAGE 2: BiLSTM │
                                                │ Sequence Engine  │
                                                │ (78 Input Feats) │
                                                └────────┬─────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────┐
                                                │ Verified Complex │
                                                │ Intrusion Alert  │
                                                └──────────────────┘
```

---

## 2. System Architecture

The inspection pipeline operates in two distinct, coordinated stages:

### Stage 1: Line-Rate Tabular Classification (XGBoost Fast-Path)
* **Feature Extraction (`extract_features_v7`):** Extracts 70 deterministic features per flow:
  * **42 Core Flow Features:** Inter-arrival timing statistics, forward/backward packet sizes, flow duration, byte ratios, TCP window dynamics, TCP flag combinations (SYN, ACK, FIN, RST, PSH, URG), and transport protocols.
  * **28 Protocol-Enriched Features:** HTTP request methods, URI lengths, status codes, TLS server name indication (SNI) entropy, cipher suite lengths, DNS query types, response lengths, and record counts.
* **Classifier:** 6-Class Multi-Source Supervised XGBoost Classifier (`models/baseline/ids_model_v10c_baseline.pkl`).
* **Decision Policy ($T = 0.84$):**
  * $P(\text{Attack}) < 0.84 \longrightarrow$ Categorized as **Benign** (zero alert latency).
  * $P(\text{Attack}) \ge 0.84 \land \text{Class} \in \{\text{DoS}, \text{DDoS}\} \longrightarrow$ Immediate alert generated (**Fast-Path**; skips Stage 2 to prevent volumetric denial-of-service processing bottlenecks).
  * $P(\text{Attack}) \ge 0.84 \land \text{Class} \in \{\text{WebAttack}, \text{Bot}, \text{Infiltration}\} \longrightarrow$ Dispatched to the Stage 2 sliding window for temporal verification.

### Stage 2: Temporal Deep Inspection (`IPBuffer` + BiLSTM Engine)
* **Sliding Window Buffer (`IPBuffer`):** Maintains stateful sliding windows of the last $L = 40$ flows for each unique Source IP.
* **8 Temporal Behavioral Features:**
  * Inter-flow beaconing interval mean and variance (detects C&C heartbeat frequencies).
  * Destination IP Shannon entropy (detects lateral movement and IP sweeps).
  * Destination port concentration ratio (detects vertical port scanning).
  * DNS query rate per minute (detects fast-flux domain resolution and data exfiltration).
  * TLS SNI reuse factor and payload size variance.
* **Classifier:** Bidirectional LSTM (`BiLSTM(64) -> Dropout(0.3) -> LSTM(32) -> Dense(3, Softmax)`) with 78 concatenated inputs (70 base + 8 behavioral), outputting probabilities for `[Volumetric, WebAttack, Botnet]`.
* **Stage-2 Operational Note:** The Stage-2 BiLSTM weights (`pipeline/lstm_best.keras`) were preserved from previous stable iterations without retraining on v10c multi-source data. Volumetric threats (DoS/DDoS) are handled directly at line-rate by the Stage-1 XGBoost fast-path, while complex sequential threats are passed to the preserved BiLSTM buffer.

---

## 3. Engineering Post-Mortem: The Failure and Abandonment of CORAL

A defining milestone in this project's R&D lifecycle was the extensive evaluation and subsequent complete deprecation of **CORAL (Domain Adaptation via Covariance Alignment)**.

```
+---------------------------------------------------------------------------------------------------+
|                                 THE CORAL BLINDING EFFECT                                         |
|                                                                                                   |
|  Source (Training Domain) Covariance:  Contains both Benign and Malicious Variance                |
|  Target (Live Production) Covariance:  Contains ONLY Benign Enterprise Variance                   |
|                                                                                                   |
|  Transformation:  X_coral = X * (C_Target)^(-1/2) * (C_Source)^(1/2)                              |
|                                                                                                   |
|  Result: Malicious feature vectors are rotated into the benign subspace, obliterating            |
|          attack signatures. False Alarms drop to 0.04%, but Attack Recall collapses to 3.79%.     |
+---------------------------------------------------------------------------------------------------+
```

### The Mechanism of Failure
1. **Hypothesis:** Academic datasets (CICIDS2018) suffer from domain shift when deployed in live enterprise networks (CICIDS2017 / Suricata). Unsupervised covariance whitening was hypothesized to align feature distributions without requiring labeled live attack data.
2. **The Illusion of Success:** Initial benchmarks showed that CORAL reduced False Alarm Rate from **74.68% down to 0.04%**, seemingly resolving the domain shift issue.
3. **The Discovery of Silent Blinding:** When comprehensive end-to-end recall was measured against diverse attack vectors, detection recall had collapsed from **98.57% down to 3.79%** (over **96.2% of all cyber attacks were completely missed**).
4. **Root Cause Analysis:** Because live baseline traffic contains zero attack samples, the target covariance matrix $C_T$ reflects only benign office communication. Whitening incoming traffic by $C_T^{-1/2}$ projects high-variance attack vectors (DDoS bursts, brute force sequences) directly into the benign manifold.

### The Architectural Pivot
* **Permanent Ban on Unilateral Covariance Whitening:** All CORAL and Optimal Transport post-hoc transformation layers were permanently excised from the production stack.
* **The Multi-Source Supervised Solution:** Rather than altering mathematical coordinate spaces post-hoc, the model was trained across diverse, curated ground-truth datasets (**CICIDS2018 + CTU-13 + CICIDS2017 holdout**), enabling the tree ensembles to learn resilient, domain-invariant decision boundaries natively.

For full mathematical proofs and forensic logs, see [`docs/CORAL_POST_MORTEM.md`](docs/CORAL_POST_MORTEM.md).

---

## 4. Public Academic Datasets & Data Lineage

All training, validation, and holdout data originates strictly from publicly available, peer-reviewed academic security repositories:

| Dataset | Research Institution & Citation | Role in Project | Data Lineage & Curation |
| :--- | :--- | :--- | :--- |
| **[CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)** | Canadian Institute for Cybersecurity (UNB / AWS) | Base Multi-Class Training | Full 10-day dataset re-parsed through Suricata 7 to generate native `eve.json` records (DoS, DDoS, WebAttack, Infiltration). |
| **[CTU-13](https://www.stratosphereips.org/datasets-ctu13)** | Stratosphere Laboratory, Czech Technical University | Dedicated Botnet Training | 13 real botnet capture scenarios. Cleaned to isolate **120,352 pure malicious flows** (C&C, Exploit, SPAM, PortScan), removing background benign DNS/NTP resolver noise. |
| **[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)** | Canadian Institute for Cybersecurity (ISCX) | Zero-Shot Evaluation & Clean Holdout | **194,480 verified benign flows** strictly isolated from **Friday-WorkingHours** (background traffic after isolating LOIC attack subnets). **Thursday was eliminated** due to contaminated background Neris botnet traffic and unsegmented scanning streams. |

For detailed dataset cleaning and bias audits, see [`docs/LEAKAGE_AND_BIAS_DIAGNOSIS.md`](docs/LEAKAGE_AND_BIAS_DIAGNOSIS.md).

---

## 5. Verified Production Performance (`v10c_baseline`)

The production model bundle (`models/baseline/ids_model_v10c_baseline.pkl`) was verified on a **194,480-flow independent clean office holdout** (CICIDS2017 Friday) and a **50,000-flow multi-class attack validation suite**:

### A. False Alarm Rate (FAR) on Clean Enterprise Holdout
* **Total Evaluated Clean Flows:** $194,480$
* **Production Operating Point ($T = 0.84$):** **0.42% FAR** ($812 / 194,480$ false alarms) ✅ *(Below the enterprise ceiling of $\le 0.50\%$)*
* **High-Sensitivity Operating Point ($T = 0.80$):** **1.38% FAR** ($2,679 / 194,480$ false alarms)

### B. End-to-End Multi-Class Attack Recall

| Attack Category | Holdout Sample Size | Production Detections ($T = 0.84$) | Production Recall ($T = 0.84$) | High-Sensitivity Detections ($T = 0.80$) | High-Sensitivity Recall ($T = 0.80$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DoS (SYN / HTTP Flood / Slowloris)** | $10,000$ | $8,738$ / $10,000$ | **87.38%** | $9,443$ / $10,000$ | 94.43% |
| **DDoS (Academic Synthetic)** | $10,000$ | $7,274$ / $10,000$ | **72.74%** | $8,775$ / $10,000$ | 87.75% |
| **Web Attacks (SQLi, XSS, Brute Force)** | $10,000$ | $8,330$ / $10,000$ | **83.30%** | $9,252$ / $10,000$ | 92.52% |
| **Botnet (C&C, SPAM, Scanning)** | $10,000$ | $8,695$ / $10,000$ | **86.95%** | $9,521$ / $10,000$ | 95.21% |
| **Infiltration (Exploits, Port Scanning)** | $10,000$ | $8,666$ / $10,000$ | **86.66%** | $9,449$ / $10,000$ | 94.49% |
| **OVERALL SYSTEM RECALL** | **$50,000$** | **$41,703$ / $50,000$** | **83.41%** | **$46,440$ / $50,000$** | **92.88%** |

### C. Zero-Shot Generalization (CICIDS2017 Friday Live LOIC DDoS)
Evaluation against real-world LOIC DDoS traffic that the model was **never trained on**:
* **Zero-Shot Recall ($T = 0.70$):** **97.10%**
* **Zero-Shot Recall ($T = 0.80$):** **79.80%**
* **Zero-Shot Recall ($T = 0.84$):** **72.40%**

For the full threshold operating curve table ($T=0.50 \dots 0.95$), see [`docs/THRESHOLD_AND_OPERATING_CURVES.md`](docs/THRESHOLD_AND_OPERATING_CURVES.md).

---

## 6. Future Work: Deep Tabular-Temporal Transformer (Planned Architecture)

*Note: This architecture is a conceptual research proposal currently in active prototype design in a separate development workspace (`xgboost-v2/`). It has not yet been benchmarked or deployed in production.*

* **Proposed Architecture:** Unified PyTorch-based neural engine:
  * **MLP Flow Tokenizer:** Non-linear projection of 70 flow features into higher-dimensional token embeddings.
  * **Transformer Sequence Encoder:** Scaled dot-product attention with temporal positional encoding over sliding IP windows ($L = 40$), capturing multi-flow attack progressions without heuristic stage separation.
  * **Multi-Task Output Heads:** Simultaneous prediction for binary anomaly detection, 6-class coarse attack categorization, and fine-grained sub-attack labeling.

---

## 7. Quickstart & CLI Usage

### Prerequisites
```bash
pip install xgboost==3.2.0 scikit-learn pandas numpy tensorflow
```

### Real-Time Live Suricata Tail Mode
```bash
tail -f /var/log/suricata/eve.json | python3 pipeline/hybrid_inference.py
```

### Batch Analysis Mode
```bash
python3 pipeline/hybrid_inference.py \
    --eve /path/to/suricata_eve.json \
    --batch \
    --output ./alerts.json \
    --xgb_model models/baseline/ids_model_v10c_baseline.pkl \
    --lstm_model pipeline/lstm_best.keras
```

---

## 8. Repository Structure

```
.
├── config/                     # Suricata feature extraction configurations
│   ├── feature_config.json
│   └── suricata_feature_extract.yaml
├── docs/                       # Technical reports and forensic investigations
│   ├── PROJECT_MASTER.md       # Master architecture logbook
│   ├── TEKNIK_SPEC.md          # 70-feature mathematical specification v3.0
│   ├── CORAL_POST_MORTEM.md    # Analysis of covariance whitening failure
│   ├── LEAKAGE_AND_BIAS_DIAGNOSIS.md # LOIC temporal leakage & dataset bias audit
│   └── THRESHOLD_AND_OPERATING_CURVES.md # Full ROC operating curve matrix
├── models/
│   ├── baseline/               # 👑 Production Deployment Binaries
│   │   └── ids_model_v10c_baseline.pkl
│   ├── ids_model_v10_final.pkl
│   ├── ids_model_v10b_final.pkl
│   ├── ids_model_v10c_final.pkl
│   └── ids_model_v10d_final.pkl
├── pipeline/                   # Core inference and training engines
│   ├── hybrid_inference.py     # Production inference engine
│   ├── ip_buffer.py            # Sliding IP window buffer & behavioral metrics
│   ├── lstm_best.keras         # Stage 2 LSTM weights
│   ├── features.py             # Feature computation helper routines
│   └── trainv10c.py            # Multi-source training script
├── scripts/                    # Evaluation, benchmark, and audit tools
│   ├── evaluate_hybrid_e2e_attacks.py
│   ├── evaluate_large_scale_benign.py
│   └── compare_all_models.py
└── archive/                    # Historical models, experimental logs (Git LFS)
```

---

## 9. Open-Source License & Copyleft Terms

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

* **Open-Source Guarantee:** You are free to inspect, run, modify, and distribute this software and its associated model artifacts.
* **Copyleft Requirement:** In accordance with the GNU GPLv3, any derivative works, modified versions, or larger projects built using this software or its components **must also be released as free and open-source software under the terms of the GNU General Public License v3.0**.
* See the full license text in the [`LICENSE`](LICENSE) file.
