# Technical Specification — Hybrid Suricata NIDS (XGBoost + BiLSTM)

---

> **Version:** 3.0  
> **Last Updated:** August 17, 2026  
> **Production Baseline:** `v10c_baseline` (`models/baseline/ids_model_v10c_baseline.pkl`)  
> **Inference Engine:** `pipeline/hybrid_inference.py`  
> **License:** GNU General Public License v3.0 (GPLv3)  

---

## 1. Architectural Specification

### 1.1 Scope & Pipeline Logic
The system processes streaming Suricata `eve.json` records through a two-stage hybrid inspection pipeline:
* **Stage 1 (XGBoost Fast-Path):** Single-pass 70-feature multi-class classification (`ids_model_v10c_baseline.pkl`). Benign flows are filtered at threshold $T=0.84$; volumetric DoS/DDoS attacks trigger immediate alerts without sequential overhead; slow/multi-hop threats (WebAttack, Infiltration, Bot) are routed to Stage 2.
* **Stage 2 (BiLSTM Sequence Engine):** Sliding-window buffer (`IPBuffer`, $L=40$) computing 8 behavioral sequence features (78 total inputs), classifying into `[Volumetric, WebAttack, Botnet]`.

### 1.2 Data Flow Diagram

```
Suricata eve.json (~68 GB)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  FlowEnrichment (Single-Pass Cache)                 │
│  Correlates flow_id with HTTP/TLS/DNS events        │
│  Memory Bounds: Max 200k entries, 300s timeout      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  extract_features_v7()  →  70 Features              │
│  42 Core Flow Dynamics + 28 Protocol-Enriched       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: XGBoost Multi-Source (v10c Baseline)      │
│  Model: models/baseline/ids_model_v10c_baseline.pkl │
│  Probability Vector: [Benign, DoS, DDoS, Web, Infil,│
│                       Bot]                          │
│                                                     │
│  Production Operating Threshold: T = 0.84           │
│                                                     │
│  Decision Logic:                                    │
│    P(Attack) < 0.84 OR Pred == Benign               │
│      → Ignore (Pass)                                │
│    Pred ∈ {DoS, DDoS}                               │
│      → Fast-Path Alert (Skip Stage 2)               │
│    Pred ∈ {WebAttack, Infiltration, Bot}            │
│      → Enqueue to IPBuffer                          │
└─────────────────────────────────────────────────────┘
        │ (WebAttack / Infiltration / Bot flows)
        ▼
┌─────────────────────────────────────────────────────┐
│  IPBuffer (Per-Source-IP Deque, Window L = 40)      │
│  Triggered on window full → Compute 8 Behavioral    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  compute_ip_window_features()  →  8 Behavioral      │
│  beacon_mean/var, dst_ip_entropy, dns_rate,         │
│  uri_entropy, same_dst_port_ratio, tls_reuse,       │
│  payload_size_variance                              │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Concatenation: 70 Core + 8 Behavioral = 78 Inputs  │
│  Reshape: (1, 40, 78) → Input to LSTM Engine        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: BiLSTM Multi-Class Sequence Model         │
│  BiLSTM(64) → LSTM(32) → Dense(32) → Dense(3)       │
│  Classes: Volumetric(0), WebAttack(1), Bot(2)       │
│  Confidence Threshold: 50%                          │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Alert Notification (JSON Streaming):
  {"src_ip": ..., "label": ..., "confidence": ..., "stage": ..., "timestamp": ...}
```

---

## 2. Verified Performance Metrics (`v10c_baseline`)

### 2.1 Holdout Operating Characteristic ($194,480$ Clean Enterprise Flows)

| Threshold ($T$) | Benign Holdout FAR (%) | DoS Recall (%) | DDoS Recall (%) | Zero-Shot LOIC DDoS (%) | Botnet Recall (%) | Web Attack Recall (%) | Infiltration Recall (%) | Overall System Recall (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.70** | 4.05% | 98.42% | 97.15% | 97.10% | 98.85% | 98.04% | 98.70% | **98.23%** |
| **0.75** | 2.87% | 97.21% | 94.93% | 91.80% | 97.99% | 96.53% | 97.60% | **96.85%** |
| **0.80** | 1.38% | 94.43% | 87.75% | 79.80% | 95.21% | 92.52% | 94.49% | **92.88%** |
| **0.84** ⭐ | **0.42%** | **87.38%** | **72.74%** | **72.40%** | **86.95%** | **83.30%** | **86.66%** | **83.41%** |

---

## 3. Engineering Post-Mortems

### 3.1 CORAL Domain Adaptation Deprecation
* Covariance whitening ($C_T^{-1/2}$) of streaming flows against target enterprise baselines rotated high-variance attack vectors directly into the benign subspace, reducing true recall from **98.57% down to 3.79%**.
* Deprecated permanently in favor of **Multi-Source Supervised Training** across heterogeneous capture datasets.

### 3.2 Single-IP DDoS Memorization (Rejection of `v10d`)
* Training on raw Friday LOIC captures resulted in artificial 100% test scores due to packet-level leakage from a single attacker IP (`172.16.0.1`).
* Evaluation under true **Zero-Shot conditions** confirmed that `v10c_baseline` generalizes to unseen LOIC attacks with **72.40% - 97.10% recall**.

---

## 4. Execution Commands

### Live Suricata Inspection:
```bash
python3 pipeline/hybrid_inference.py --eve /var/log/suricata/eve.json
```

### Batch File Processing:
```bash
python3 pipeline/hybrid_inference.py --eve test_traffic.json --batch --output alerts.json
```
