# Teknik Spec Dokümanı — Hibrit IDS (XGBoost + LSTM)

> **Versiyon:** 3.0  
> **Son Güncelleme:** 16 Ağustos 2026  
> **Durum:** v10c Baseline = Üretim (Production), v7/v8 = Arşivlendi, CORAL = Kalıcı Olarak Devre Dışı  
> **Hedef Kitle:** Tez danışmanı, proje jürisi, güvenlik mühendisleri

---

## 1. Sistem Mimari Genel Bakış

### 1.1 Amaç ve Üretim Durumu
Suricata IDS'in `eve.json` çıktılarını gerçek zamanlı işleyerek iki aşamalı (hibrit) bir saldırı tespit sistemi sunmak:
- **Aşama 1 (XGBoost):** 70 öznitelik ile 6-sınıflı multiclass sınıflandırma (`ids_model_v10c_baseline.pkl`). Benign sınıfı eşikleme ($T=0.84$) ile filtrelenir; DoS/DDoS sınıfları doğrudan alarm üretir; WebAttack/Infiltration/Bot sınıfları davranışsal analiz için IPBuffer ve LSTM aşamasına gönderilir.
- **Aşama 2 (LSTM):** 78 öznitelik (70 core + 8 behavioral) ile 3-sınıflı sınıflandırma (Volumetric / WebAttack / Bot).

### 1.2 Mimari Şema

```
Suricata eve.json (~68 GB)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  FlowEnrichment (single-pass cache)                 │
│  flow_id bazında HTTP/TLS/DNS eventlerini eşleştirir │
│  Bellek: max 200K kayıt, 300s timeout ile temizlik   │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  extract_features_v7()  →  70 öznitelik             │
│  42 core + 28 enriched (tcp/http/tls/dns)           │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 1: XGBoost Multi-Source (v10c Baseline: 33MB) │
│  Model: models/baseline/ids_model_v10c_baseline.pkl │
│  Çıktı: 6 olasılık vektörü                         │
│    [Benign, DoS, DDoS, WebAttack, Infiltration, Bot]│
│                                                      │
│  Üretim Eşiği: 0.84 (FAR = %0.42 @ 192k Clean Flows)│
│                                                      │
│  Karar Mantığı:                                     │
│    prob_attack < 0.84 VEYA xgb_pred == Benign(0)    │
│      → Yok say (zararsız)                           │
│    xgb_pred == DoS(1) VEYA DDoS(2)                  │
│      → DoS/DDoS alarmı üretilir, LSTM ATLANIR       │
│    xgb_pred == WebAttack(3) VEYA Infiltration(4)    │
│      VEYA Bot(5)                                    │
│      → IPBuffer'a gönderilir                        │
└─────────────────────────────────────────────────────┘
        │ (WebAttack/Infiltration/Bot akışları)
        ▼
┌─────────────────────────────────────────────────────┐
│  IPBuffer — src_ip başına deque (40 akış)           │
│  40 akış dolunca → behavioral öznitelik hesapla     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  compute_ip_window_features()  →  8 behavioral      │
│  beacon_mean/std, dst_ip_entropy, dns_per_min,      │
│  uri_entropy, same_dst_port_ratio, tls_sni_reuse,   │
│  payload_size_variance                              │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Concat: 70 core + 8 behavioral = 78 öznitelik     │
│  reshape(1, 40, 78) → LSTM girdisi                 │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 2: LSTM 3-Sınıflı Multiclass                │
│  BiLSTM(64) → LSTM(32) → Dense(32) → Dense(3)     │
│  Tahmin: Volumetric(0) / WebAttack(1) / Bot(2)     │
│  Güven eşiği: %50 (altında → "Generic Attack")     │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Alarm Çıktısı (JSON):
  {src_ip, label, confidence, stage, timestamp}
```

---

## 2. Üretim Modeli Performansı (v10c Baseline)

### 2.1 Eşik Tarama ve İşletim Noktaları ($192.942$ Doğrulanmış Temiz Ofis Akışı)

| Eşik ($T$) | FAR (Temiz Ağ) | DoS Recall | DDoS Recall (Akademik) | LOIC DDoS (Cuma Zero-Shot) | Bot Recall | WebAttack Recall | Infiltration Recall | Genel Ortalama Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.70** | %4.06 | %98.40 | %97.20 | %97.10 | %98.90 | %98.20 | %98.50 | **%98.23** |
| **0.75** | %2.88 | %97.20 | %94.90 | %91.80 | %98.00 | %96.80 | %97.40 | **%96.85** |
| **0.80** | %1.38 | %94.40 | %87.80 | %79.80 | %95.20 | %92.50 | %94.50 | **%92.88** |
| **0.84** ⭐ | **%0.42** | **%87.40** | **%72.70** | **%72.40** | **%87.00** | **%83.30** | **%86.70** | **%83.41** |

---

## 3. Mimari Post-Mortem ve İspatlar

### 3.1 CORAL Domain Adaptation Neden Çöktü ve İptal Edildi?
* **Matematiksel Arka Plan:** CORAL, hedef domain'in ($C_T$) ve kaynak domain'in ($C_S$) kovaryans matrislerini hesaplayıp $A = C_S^{-1/2} C_T^{1/2}$ dönüşümü ile girdileri döndürmektedir.
* **Kırılma Noktası:** $C_T$ sadece temiz ofis trafiğinden hesaplandığı için, gelen verideki çok-değişkenli saldırı sinyalleri "temiz ofis" kovaryansına doğru ezilmektedir (whitening).
* **Kanıt:** %98.57 recall değerine sahip çıplak XGBoost modeli, CORAL transformundan geçtiğinde **recall %3.79'a çökmüştür**. Sistem sahte bir güven hissi vererek saldırıların %96.21'ini kaçırmaktadır. Bu nedenle CORAL tamamen yürürlükten kaldırılmıştır.

### 3.2 Tek Tip DDoS Ezberlemesi (v10d Neden Elendi?)
* V10d denemesinde Cuma DDoS LOIC akışlarının ($40.000$ adet) eğitime eklenmesiyle elde edilen %100 test skoru araştırılmış; aynı sürekli LOIC saldırısının (`172.16.0.1 -> 192.168.10.50`) test setine sızması (data leakage) nedeniyle modelin genel DDoS kavramı yerine tek bir aracın mikrosaniyelik GET paketlerini ezberlediği ve diğer saldırı sınıflarının recall'unu %6-16 oranında düşürdüğü tespit edilmiştir.
* V10c Baseline'ın hiç görmediği bu saldırıyı zero-shot olarak %72.4 - %97.1 oranında yakalayabilmesi, v10c'nin gerçek genelleme yeteneğini kanıtlamıştır.

---

## 4. Kullanım ve Komutlar

### Üretim Modeli ile Canlı Dinleme:
```bash
python3 pipeline/hybrid_inference.py --eve /var/log/suricata/eve.json
```

### Üretim Modeli ile Batch Dosya Taraması:
```bash
python3 pipeline/hybrid_inference.py --eve test_traffic.json --batch --output alerts.json
```
