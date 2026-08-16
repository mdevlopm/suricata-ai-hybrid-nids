# Suricata AI Hibrit Saldırı Tespit Sistemi (Hybrid NIDS)
### Çok Kaynaklı Süpervize XGBoost + Zamansal LSTM Ağ Güvenliği Çözümü

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Suricata EVE](https://img.shields.io/badge/Suricata-EVE_JSON_Native-orange.svg)](https://suricata.io)
[![Production Baseline](https://img.shields.io/badge/Production-v10c_Baseline-green.svg)](models/baseline/)
[![Status](https://img.shields.io/badge/FAR-0.42%25-brightgreen.svg)](docs/THRESHOLD_AND_OPERATING_CURVES.md)
[![Recall](https://img.shields.io/badge/Recall-83.41%25-success.svg)](docs/THRESHOLD_AND_OPERATING_CURVES.md)

---

## 📌 1. Proje Genel Bakış

Bu proje, geleneksel imza tabanlı IDS (Suricata) kurallarının yetersiz kaldığı sıfır-gün (zero-day), çok aşamalı ve gizli ağ saldırılarını gerçek zamanlı olarak tespit etmek için tasarlanmış **2 Aşamalı Hibrit Yapay Zeka Saldırı Tespit Sistemi'dir (Hybrid NIDS)**.

Sistem, Suricata'nın `eve.json` çıktılarını doğrudan dinler; 70 temel ve zenginleştirilmiş akış özniteliğini çıkararak milisaniyeler içinde yüksek hacimli saldırıları (DoS/DDoS) süzer, yavaş ve zamana yayılan davranışsal saldırıları (Web Attack, Botnet, Infiltration) ise 2. aşamadaki LSTM zaman penceresine ileterek doğrular.

---

## 🏗️ 2. Sistem Mimarisi

```mermaid
flowchart TD
    A[Suricata eve.json Canlı Akışı] --> B[FlowEnrichment: 70 Öznitelik Çıkarımı]
    B --> C[1. Aşama: XGBoost Multi-Source Sınıflandırıcı]
    
    C -- "P(Attack) < 0.84" --> D[Temiz Trafik / Benign]
    C -- "P(Attack) >= 0.84" --> E{Saldırı Türü Nedir?}
    
    E -- "DoS veya DDoS (Hacimsel)" --> F[🚨 Anında Alarm / Engelleme (XGBoost Fast-Path)]
    E -- "WebAttack / Bot / Infiltration" --> G[IPBuffer: 40 Akışlık Zaman Penceresi]
    
    G --> H[8 Davranışsal Öznitelik Ekleme: Entropi, Beaconing, DNS Sıklığı]
    H --> I[2. Aşama: LSTM Zamansal Sınıflandırıcı (78 Öznitelik)]
    
    I -- "LSTM Güven >= 0.60" --> J[🚨 Doğrulanmış Davranışsal Saldırı Alarmı]
    I -- "LSTM Güven < 0.60" --> K[⚠️ Genel Anomali / Generic Attack]
```

---

## 📊 3. Üretim Modeli Performans Metrikleri (`v10c_baseline`)

Sistem, **192.942 akışlık doğrulanmış bağımsız temiz ofis trafiği** ve **25.000 akışlık saldırı kümeleri** üzerinde uçtan uca test edilmiştir:

### A. Yanlış Alarm Oranı (FAR - False Alarm Rate)
* **Test Edilen Temiz Akış:** $192.942$
* **Üretilen Yanlış Alarm (FP):** **$805$ akış**
* **Resmi Üretim FAR ($T=0.84$):** **%0.42** ✅ *(Kurumsal SOC hedefi <%0.50 sağlanmıştır)*

### B. Uçtan Uca Saldırı Yakalama Oranları (Recall)

| Saldırı Sınıfı | Test Akış Sayısı | Yakalanan Alarm | E2E Recall ($T=0.84$) | $T=0.80$ Recall |
| :--- | :---: | :---: | :---: | :---: |
| **DoS (SYN/HTTP Flood)** | $5.000$ | $4.302$ | **%86.04** | %89.50 |
| **DDoS (Akademik)** | $5.000$ | $3.683$ | **%73.66** | %76.20 |
| **Web Attack (SQLi, XSS, Brute)** | $5.000$ | $4.189$ | **%83.78** | %86.80 |
| **Botnet (C&C, SPAM, Scan)** | $5.000$ | $4.319$ | **%86.38** | %88.90 |
| **Infiltration (Port, Exploit)** | $5.000$ | $4.360$ | **%87.20** | %89.40 |
| **GENEL SALDIRI RECALL** | **$25.000$** | **$20.853$** | **%83.41** | **%86.16** |

### C. Zero-Shot Canlı Saldırı Genellemesi (CICIDS2017 Cuma LOIC DDoS)
Modelin eğitimde **HİÇ GÖRMEDİĞİ** gerçek dünya Cuma LOIC taşkınındaki başarısı:
* **Recall ($T=0.70$):** **%97.10**
* **Recall ($T=0.80$):** **%79.80**
* **Recall ($T=0.84$):** **%72.40**

---

## 🔬 4. Model Evrimi ve Adli Analizler (Engineering Post-Mortems)

| Sürüm | Temel Özellik / Değişiklik | FAR (%) | Recall (%) | Durum / Karar Raporu |
| :--- | :--- | :---: | :---: | :--- |
| **v6 / v7** | Sentetik CICIDS2018 (70 Öznitelik) | %74.68 | %98.20 | 🔴 Alan kayması (Domain shift) yüzünden çöktü (Arşivlendi). |
| **v8 (CORAL)** | Kovaryans Hizalama (Domain Adaptation) | %0.04 | **%3.79** | 🔴 **CORAL Felaketi:** Saldırı sinyalini temiz uzaya izdüşürerek körleşme yarattı. [Raporu Oku](docs/CORAL_POST_MORTEM.md) |
| **v9 Ailesi** | MCFP ve Kontamine Veri Seti | %99.90 | — | 🔴 Kontaminasyon yüzünden silindi. |
| **v10** | Ham CTU-13 + CICIDS2018 | %17.00 | %94.00 | 🟡 CTU-13 içindeki arka plan DNS trafiği gürültü yarattı. |
| **v10b** | Temizlenmiş CTU-13 Bot (120k) | %0.85 | %92.00 | 🟢 Bot gürültüsü temizlendi, AWS resolver sızıntısı izole edildi. |
| **v10c (Baseline)** | Saf Bot (CTU-13 + 2018) + Çok Kaynaklı Süpervize | **%0.42** | **%83.41** | 👑 **NİHAİ ÜRETİM MODELİ** |
| **v10d** | LOIC Cuma Saldırısı Eklemeli | %0.21 | %73.79 | 🟡 **Sızıntı Tespiti:** Tekil IP ezberlediği için elendi. [Raporu Oku](docs/LEAKAGE_AND_BIAS_DIAGNOSIS.md) |

---

## 📁 5. Dizin Yapısı

```
.
├── config/                     # Model eşikleri, sınıf haritaları ve port tabloları
│   └── feature_config.json
├── docs/                       # Kapsamlı Teknik Raporlar & Adli İncelemeler
│   ├── PROJECT_MASTER.md       # Proje ana mimari kütüğü
│   ├── TEKNIK_SPEC.md          # Teknik şartname ve 70-öznitelik matrisi v3.0
│   ├── CORAL_POST_MORTEM.md    # CORAL çöküşü ve kovaryans beyazlatma analizi
│   ├── LEAKAGE_AND_BIAS_DIAGNOSIS.md # v10d LOIC sızıntısı ve veri temizleme
│   └── THRESHOLD_AND_OPERATING_CURVES.md # 0.50 - 0.99 tam ROC işletim tablosu
├── models/
│   ├── baseline/               # 👑 Üretim Modelleri
│   │   └── ids_model_v10c_baseline.pkl # (33 MB)
│   ├── ids_model_v10_final.pkl
│   ├── ids_model_v10b_final.pkl
│   ├── ids_model_v10c_final.pkl
│   └── ids_model_v10d_final.pkl
├── pipeline/                   # Çıkarım ve Eğitim Motorları
│   ├── hybrid_inference.py     # 🚀 Üretim Çıkarım Motoru (XGBoost + LSTM)
│   ├── ip_buffer.py            # Kayar IP pencere tamponu ve 8 davranışsal öznitelik
│   ├── lstm_best.keras         # 2. Aşama LSTM ağırlıkları
│   ├── trainv10c.py            # Üretim modeli eğitim scripti
│   └── trainv8.py              # 70 öznitelik çıkarıcı çekirdek mantık
├── scripts/                    # Doğrulama, ROC tarama ve adli test araçları
│   ├── evaluate_hybrid_e2e_attacks.py
│   ├── verify_v10c_holdout.py
│   └── temporal_slice_audit.py
└── archive/                    # Eski / Tarihi modeller ve scriptler (Git LFS)
```

---

## 🚀 6. Hızlı Başlangıç ve Çalıştırma

### Gereksinimler
```bash
pip install xgboost==3.2.0 scikit-learn pandas numpy
```

### Canlı veya Toplu Suricata Loglarını Analiz Etme
```bash
# Toplu mod (Batch Inference)
python3 pipeline/hybrid_inference.py --eve /path/to/eve.json --batch --output /tmp/alerts.json

# Canlı tail modu (Suricata canlı dinleme)
tail -f /var/log/suricata/eve.json | python3 pipeline/hybrid_inference.py
```

---

## 📜 Lisans ve Mülkiyet
Bu depo özel (private) bir araştırma ve geliştirme projesidir. Tüm hakları saklıdır.
