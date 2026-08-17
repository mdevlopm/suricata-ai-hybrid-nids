# Threshold & ROC Operating Curves Specification (Model `v10c_baseline`)

---

## 1. Executive Summary

This document specifies the exact empirical operating characteristics of the production IDS model bundle (`models/baseline/ids_model_v10c_baseline.pkl`).

All metrics are measured against:
* **Clean Enterprise Holdout Set:** $194,480$ verified benign enterprise flows from CICIDS2017 Friday.
* **Multi-Class Attack Holdout Set:** $50,000$ independent attack flows ($10,000$ per attack category).

---

## 2. Complete Operating Matrix ($T = 0.50 \dots 0.95$)

| Decision Threshold ($T$) | Benign Holdout FAR (%) | False Alarms / Total ($N=194,480$) | DoS Recall (%) | DDoS Recall (%) | Web Attack Recall (%) | Botnet Recall (%) | Infiltration Recall (%) | Overall Attack Recall (%) | Operational Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50** | 8.81% | 17,136 / 194,480 | 99.62% | 99.38% | 99.64% | 99.78% | 99.71% | **99.63%** | High Alert (Noisy) |
| **0.60** | 6.23% | 12,113 / 194,480 | 99.27% | 98.86% | 99.11% | 99.52% | 99.33% | **99.22%** | High Sensitivity |
| **0.70** | 4.05% | 7,870 / 194,480 | 98.42% | 97.15% | 98.04% | 98.85% | 98.70% | **98.23%** | Moderate Filtering |
| **0.75** | 2.87% | 5,584 / 194,480 | 97.21% | 94.93% | 96.53% | 97.99% | 97.60% | **96.85%** | Pre-Production |
| **0.78** | 2.01% | 3,901 / 194,480 | 95.90% | 91.55% | 94.72% | 96.64% | 96.06% | **94.97%** | Strict Filtering |
| **0.80** | **1.38%** | 2,679 / 194,480 | 94.43% | 87.75% | 92.52% | 95.21% | 94.49% | **92.88%** | 🛡️ High-Recall Mode |
| **0.82** | 0.83% | 1,622 / 194,480 | 91.74% | 81.66% | 89.06% | 92.30% | 91.45% | **89.24%** | Balanced Mode |
| **0.84** | **0.42%** | **812 / 194,480** | **87.38%** | **72.74%** | **83.30%** | **86.95%** | **86.66%** | **83.41%** | 👑 **PRODUCTION BASELINE** |
| **0.86** | 0.16% | 317 / 194,480 | 80.09% | 58.81% | 71.57% | 79.89% | 78.39% | **73.75%** | Ultra-Low False Alarm |
| **0.88** | 0.04% | 84 / 194,480 | 56.69% | 28.43% | 42.06% | 70.03% | 55.71% | **50.58%** | High Suppression |
| **0.90** | 0.01% | 29 / 194,480 | 35.47% | 11.93% | 21.81% | 57.65% | 37.44% | **32.86%** | Critical-Only |
| **0.95** | 0.00% | 9 / 194,480 | 19.98% | 7.71% | 12.28% | 37.30% | 26.53% | **20.76%** | Near-Zero Alarms |

---

## 3. Production Operating Recommendation

* **Default Production Threshold ($T = 0.84$):**
  * Recommended for enterprise operations where false positives interrupt security operations center (SOC) workflows.
  * Achieves **0.42% FAR** (less than 1 false positive per 230 flows) with an overall attack recall of **83.41%**.
* **High-Threat / Active Incident Threshold ($T = 0.80$):**
  * Recommended during suspected breaches or active reconnaissance alerts.
  * Achieves **92.88% Overall Recall** (with a manageable 1.38% FAR).
