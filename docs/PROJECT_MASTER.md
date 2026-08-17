# Hybrid Suricata NIDS — Master Architecture & Project Specification

---

**Last Updated:** August 17, 2026  
**Production Model:** `models/baseline/ids_model_v10c_baseline.pkl`  
**Inference Engine:** `pipeline/hybrid_inference.py` (Default: `v10c_baseline`, Threshold: `0.84`)  
**License:** GNU General Public License v3.0 (GPLv3)  

---

## 1. Project Purpose & Production Status

A two-stage hybrid Network Intrusion Detection System (NIDS) operating directly on native Suricata `eve.json` flow telemetry, combining **Multi-Source Supervised XGBoost (Stage 1)** and a **Sliding-Window Temporal BiLSTM (Stage 2)**.

### Verified Production Metrics (`v10c_baseline` @ $T=0.84$):
* **False Alarm Rate (FAR) on 194,480 Verified Clean Flows:** **0.42%** ($812$ false alarms)
* **High-Sensitivity Operating FAR (@ $T=0.80$):** **1.38%** ($2,679$ false alarms)
* **Cross-Dataset Attack Recall ($N=50,000$ Holdout Flows):**
  * **DoS (SYN/HTTP Floods):** **87.38%** ($8,738 / 10,000$) @ $T=0.84$ | **94.43%** @ $T=0.80$
  * **DDoS (Synthetic Academic):** **72.74%** ($7,274 / 10,000$) @ $T=0.84$ | **87.75%** @ $T=0.80$
  * **Web Attacks (SQLi, XSS, Brute Force):** **83.30%** ($8,330 / 10,000$) @ $T=0.84$ | **92.52%** @ $T=0.80$
  * **Botnet (C&C, SPAM, PortScan):** **86.95%** ($8,695 / 10,000$) @ $T=0.84$ | **95.21%** @ $T=0.80$
  * **Infiltration (Exploits, Scanning):** **86.66%** ($8,666 / 10,000$) @ $T=0.84$ | **94.49%** @ $T=0.80$
  * **OVERALL SYSTEM RECALL:** **83.41%** ($41,703 / 50,000$) @ $T=0.84$ | **92.88%** ($46,440 / 50,000$) @ $T=0.80$
* **Zero-Shot Generalization (CICIDS2017 Friday LOIC DDoS):** **72.40%** (@ $T=0.84$) / **97.10%** (@ $T=0.70$)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Suricata IDS Telemetry                                             │
│  eve.json → flow, http, tls, dns records                             │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  extract_features_v7()                                              │
│  70 Features: 42 core (timing, packets, bytes, ports, protocols)    │
│             + 28 protocol-enriched (HTTP, TLS, DNS, TCP flags)      │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: XGBoost 6-Class Multi-Source (v10c Baseline: 33 MB)       │
│  Classes: Benign(0), DoS(1), DDoS(2), WebAttack(3),                 │
│           Infiltration(4), Bot(5)                                   │
│  P(Attack) = 1.0 - P(Benign)                                        │
│  Production Threshold: 0.84                                         │
│                                                                     │
│  P(Attack) < 0.84 ───────────→ Benign (Ignored)                     │
│  Class ∈ {DoS, DDoS} ────────→ Immediate Fast-Path Alert (Skip LSTM)│
│  Class ∈ {WebAttack, Infil, Bot} ──→ Forward to IPBuffer            │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  IPBuffer (Sliding Window: L = 40 flows per unique Source IP)       │
│  compute_ip_window_features() → 8 Behavioral Sequence Features      │
│  (beacon_mean/var, dst_ip_entropy, dns_query_rate, uri_entropy,     │
│   same_dst_port_ratio, tls_sni_reuse, payload_variance)             │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Concatenation: 70 core + 8 behavioral = 78 Feature Vector          │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: BiLSTM 3-Class Sequence Classifier                        │
│  BiLSTM(64) → LSTM(32) → Dense(3) → Softmax                         │
│  Classes: Volumetric(0), WebAttack(1), Bot(2)                       │
│  Input Shape: (1, 40, 78)                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Engineering Decisions & Post-Mortems

### 3.1 Total Deprecation of CORAL Domain Adaptation
* **Failure Analysis:** Unsupervised covariance whitening using benign enterprise traffic ($C_T^{-1/2}$) rotated multi-variate attack signatures into the benign subspace, reducing False Alarm Rate from 74% to 0.04% by blinding the model (Recall collapsed from **98.57% to 3.79%**).
* **Permanent Mandate:** Unilateral covariance whitening and post-hoc feature rotations are permanently banned from the pipeline.
* **Resolution:** Replaced with true **Multi-Source Supervised Training** across CICIDS2018, CTU-13, and CICIDS2017 holdout data.

### 3.2 Botnet & Baseline Data Sanitation
* Eliminated background recursive DNS resolver traffic (`Port 53`), AWS metadata queries (`169.254.169.254`), and local DHCP/NTP noise from CTU-13 and CICIDS2018 botnet classes, driving Botnet false positives from 17.0% down to 0.42%.
* Eliminated contaminated CICIDS2017 Thursday data due to background Neris botnet noise.

---

## 4. Model Lineage

| Model Version | Architecture & Training Scope | Clean FAR | Attack Recall | Status |
| :--- | :--- | :---: | :---: | :--- |
| **v7** | Single-Source CICIDS2018 | 0.96% | 98.20% (Synthetic) | Archived (`archive/models/ids_model_v7.onnx`) |
| **v8** | Integrated CORAL Transformation | 0.04% (Fake) | **3.79% (Failure)** | Deprecated & Archived (`archive/models/ids_model_v8_final.pkl`) |
| **v9** | MCFP Contaminated | >99% | — | Failed & Deleted |
| **v10c Baseline** | **Multi-Source Supervised (Clean Bot + Holdout Friday)** | **0.42%** | **83.41% - 92.88%** | 👑 **OFFICIAL PRODUCTION BASELINE** |
