# CORAL (Correlation Alignment) Post-Mortem & Fail-Safe Report

---

## 1. Executive Summary

During several months of development, **CORAL (Domain Adaptation via Covariance Alignment)** was utilized to bridge the domain shift between synthetic academic training data (CICIDS2018) and live network telemetry (CICIDS2017 / Suricata).

CORAL initially appeared highly effective by driving the operational False Alarm Rate (FAR) from **74.68% down to 0.04%**. However, comprehensive cross-dataset recall verification revealed a critical failure: **real attack detection recall collapsed from 98.57% down to 3.79% (missing 96.21% of all active cyber attacks)**.

This document details the mathematical failure mechanism, explains the "silent blinding" effect on tree ensembles, and records the permanent design ban on unilateral post-hoc covariance whitening.

---

## 2. Mathematical Foundation of CORAL

CORAL seeks to minimize the second-order statistical distance between the covariance matrix of the source training domain ($C_S$) and the target live enterprise domain ($C_T$):

$$\min_{A} \| C_{\hat{S}} - C_T \|_F^2 = \min_{A} \| A^T C_S A - C_T \|_F^2$$

The linear transformation matrix $A$ is derived by whitening the source distribution and re-coloring it with the target covariance:

$$x_{\text{coral}} = x \cdot C_T^{-1/2} \cdot C_S^{1/2}$$

---

## 3. Root Cause of Mathematical Failure

1. **Absence of Malicious Variance in the Target Domain:**
   Target domain telemetry collected from live office networks contains exclusively benign background traffic. Consequently, the target covariance matrix $C_T$ contains zero malicious variance components.

2. **Attack Subspace Projection (The Blinding Effect):**
   Multiplying incoming network vectors by the inverse square root of target covariance ($C_T^{-1/2}$) forcibly projects multi-variate anomalous attack clusters (DDoS bursts, brute-force streams, botnet beacons) into the principal variance axes of benign office traffic.

3. **Conceptual Visualization:**
   ```
   [Active Attack Vector] (High variance across packet sizes & TCP flags)
                │
                ▼ (CORAL Covariance Whitening by C_Target^-1/2)
   [Projected Vector] (Mathematically rotated to match benign office covariance)
                │
                ▼
   [XGBoost Classifier] ──> "Classified as Benign" (Recall = 3.79%)
   ```

---

## 4. Empirical Evidence & Performance Comparison

| Metric | Raw Baseline (No CORAL) | With CORAL Transformation | Impact / Engineering Verdict |
| :--- | :---: | :---: | :--- |
| **Benign False Alarm Rate (FAR)** | 0.42% | 0.04% | Artificial reduction (Deceptive) |
| **DoS Recall** | **86.04%** | **1.20%** | 🔴 84.84% Attack Blindness |
| **DDoS Recall** | **73.66%** | **0.85%** | 🔴 72.81% Attack Blindness |
| **Web Attack Recall** | **83.78%** | **4.10%** | 🔴 79.68% Attack Blindness |
| **Botnet Recall** | **86.38%** | **8.12%** | 🔴 78.26% Attack Blindness |
| **Infiltration Recall** | **87.20%** | **4.68%** | 🔴 82.52% Attack Blindness |
| **OVERALL ATTACK RECALL** | **83.41%** | **3.79%** | 🔴 **79.62% COLLAPSE** |

---

## 5. Architectural Remediation

1. **Permanent Ban on Unilateral Covariance Whitening:** All CORAL, Optimal Transport (OT), and marginal mean/std adaptation layers were permanently removed from the inference engine.
2. **Multi-Source Supervised Training:** Instead of applying post-hoc mathematical transformations to data points, the model is trained across multiple ground-truth datasets (**CICIDS2018 + CTU-13 + CICIDS2017 holdout**), forcing the tree ensembles to learn domain-invariant decision boundaries natively (`ids_model_v10c_baseline.pkl`).
