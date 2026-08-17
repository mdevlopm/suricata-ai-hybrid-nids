# Data Leakage & Dataset Contamination Forensic Diagnosis

---

## 1. Executive Summary

During the model iteration cycle between `v9`, `v10b`, and `v10d`, rigorous forensic audits revealed critical dataset-level artifacts, feature leakages, and contamination:

1. **CICIDS2017 Thursday Contamination:** Initial attempts to use Thursday traffic as a benign baseline resulted in severe bias due to contaminated Neris botnet communication and unsegmented background scan streams.
2. **v10d DDoS Single-IP Leakage:** An apparent "100.0% recall" across all decision thresholds in `v10d` was discovered to be an artifact of random train/test splitting on a single attacker IP (`172.16.0.1`), where identical packet signatures leaked between partitions.
3. **Botnet DNS & Gateway Contamination:** Raw CTU-13 and CICIDS2018 botnet captures contained benign enterprise infrastructure traffic (AWS resolvers, default gateway ARP/DNS), causing standard models to misclassify benign DNS traffic as botnet activity (driving initial Bot FAR to 17.0%).

---

## 2. Forensic Audit Findings

### 2.1 The Thursday-WorkingHours Baseline Failure
* **Observation:** Thursday traffic exhibited erratic baseline shifts when tested against standard classifiers.
* **Root Cause:** Analysis of source IP traffic confirmed active Neris botnet traffic and unsegmented scanning hosts embedded inside the supposedly "clean" working-hours capture.
* **Resolution:** Thursday traffic was completely discarded as a baseline. The **194,480 holdout benign flows** were isolated exclusively from **CICIDS2017 Friday (PortScan background traffic after complete removal and quarantine of the LOIC attack subnet `172.16.0.1`)**.

### 2.2 v10d LOIC DDoS Data Leakage
* **Observation:** The `v10d` prototype showed a flat 100.0% recall on Friday DDoS even at extreme decision thresholds ($T = 0.95$).
* **Root Cause:** 99.14% of the Friday LOIC attack flows originated from a single IP (`172.16.0.1`). Standard random splitting allocated identical LOIC bursts into both the training and evaluation sets.
* **Resolution:** Evaluated the production baseline `v10c` strictly under **Zero-Shot conditions** (model never sees LOIC attack flows during training), yielding a realistic generalization recall of **72.40% (@ T=0.84)** and **97.10% (@ T=0.70)**.

### 2.3 Botnet Infrastructure Noise Filtering
* **Observation:** Initial `v10` models suffered an unacceptable 17.0% False Alarm Rate on benign office networks, triggered almost entirely by the Botnet class.
* **Root Cause:** Captured botnet PCAPs in CTU-13 included standard infrastructure flows:
  * Port 53 DNS queries to upstream campus resolvers
  * NTP synchronization requests (`123/UDP`)
  * AWS internal metadata service queries (`169.254.169.254`)
* **Resolution:** Built deterministic relabeling scripts (`scripts/ctu13_relabel_verified_pure.py`) that filtered out all non-malicious infrastructure traffic, retaining only pure C&C, SPAM, and exploit vectors (120,352 pure flows). This reduced Botnet-induced FAR from 17.0% down to **0.42%**.
