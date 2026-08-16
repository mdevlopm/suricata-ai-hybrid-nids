# Hibrit NIDS Projesi — Master Belge

**Son Güncelleme:** 16 Ağustos 2026  
**Konum:** `/run/media/mehmet/siber data1/ai modeli xgboost`  
**Üretim Modeli (Production):** `models/baseline/ids_model_v10c_baseline.pkl`  
**Çıkarım Motoru (Inference):** `pipeline/hybrid_inference.py` (Varsayılan Model: `v10c_baseline`, Eşik: `0.84`)

---

## 1. Proje Amacı ve Üretim Durumu

Suricata IDS'in `eve.json` çıkışından gerçek zamanlı saldırı tespiti yapan, **XGBoost (6-Sınıflı Çoklu Kaynaklı Süpervize) + LSTM** tabanlı hibrit Network Intrusion Detection System (NIDS).

### Temel Üretim Metrikleri (V10c Baseline — $T=0.84$):
- **Doğrulanmış Temiz Ağda Yanlış Alarm Oranı (FAR):** **%0.42** ($192.942$ akışta sadece $812$ alarm)
- **Doğrulanmış Temiz Ağda Yanlış Alarm Oranı (FAR @ $T=0.80$):** **%1.38**
- **Saldırı Yakalama Oranları (Recall):**
  - **DoS:** **%87.40** ($T=0.84$) / **%94.40** ($T=0.80$)
  - **DDoS (Akademik Sentetik):** **%72.70** ($T=0.84$) / **%87.80** ($T=0.80$)
  - **Gerçek Suricata LOIC DDoS (Zero-Shot Genelleme):** **%72.40 - %97.10** ($T=0.84 - 0.70$)
  - **Botnet (CTU-13 + CICIDS2018 Temizlenmiş):** **%87.00** ($T=0.84$) / **%95.20** ($T=0.80$)
  - **Web Attack:** **%83.30** ($T=0.84$) / **%92.50** ($T=0.80$)
  - **Infiltration:** **%86.70** ($T=0.84$) / **%94.50** ($T=0.80$)

---

## 2. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────┐
│  Suricata IDS                                                      │
│  eve.json → flow, http, tls, dns kayıtları                          │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  extract_features_v7()                                              │
│  70 öznitelik: 42 core (süre, paket, bayt, port, protokol, durum)  │
│             + 28 enriched (HTTP, TLS, DNS, TCP bayrakları)         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AŞAMA 1: XGBoost 6-Sınıflı Multi-Source (v10c Baseline: 33 MB)     │
│  Sınıflar: Benign(0), DoS(1), DDoS(2), WebAttack(3),               │
│            Infiltration(4), Bot(5)                                  │
│  prob_attack = 1.0 - prob[Benign]                                   │
│  Üretim Eşiği: 0.84 (Pickle içi doğrulanmış)                        │
│                                                                     │
│  prob_attack < 0.84 → Zararsız (yok say)                            │
│  argmax ∈ {1,2}      → DoS/DDoS → DOĞRUDAN ALARM (LSTM ATLA)         │
│  argmax ∈ {3,4,5}    → IPBuffer'a aktar                             │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  IPBuffer (src_ip başına deque, max 40 akış)                       │
│  compute_ip_window_features() → 8 behavioral öznitelik             │
│  (beacon_mean/std, dst_ip_entropy, dns_per_min, uri_entropy,       │
│   same_dst_port_ratio, tls_sni_reuse, payload_size_variance)       │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Concat: 70 core + 8 behavioral = 78 öznitelik                     │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AŞAMA 2: LSTM 3-Sınıflı Multiclass                                │
│  BiLSTM(64) → LSTM(32) → Dense(3) → Softmax                       │
│  Sınıflar: Volumetric(0), WebAttack(1), Bot(2)                     │
│  Window boyutu: 40 akış                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mimari Kararlar ve Tarihçe

### 3.1 CORAL Domain Adaptation'ın Tamamen Terk Edilmesi (Post-Mortem)
> **KRİTİK UYARI:** Gelecekte sisteme tekrar CORAL, whitening veya tek yönlü post-hoc kovaryans hizalama eklenmemelidir.

* **Sebep:** CORAL, yalnızca temiz ofis trafiğine göre kovaryans matrisini döndürdüğü (whitening/rotation) için, ağa giren saldırı sinyallerini de temiz ağın istatistiklerine benzetmiş ve saldırı vektörlerini silmiştir.
* **Sonuç:** Çıplak modelde **%98.57** olan saldırı yakalama oranı (Recall), CORAL uygulandığında **%3.79'a çökmüştür** (Saldırıların %96.21'i kaçırılmıştır).
* **Nihai Çözüm:** Post-hoc matematiksel dönüşümler terk edilmiş; modelin doğrudan farklı ağlardan (CTU-13, CICIDS2018, CICIDS2017) toplanan temizlenmiş zemin gerçeği ile eğitildiği **Multi-Source Supervised** mimariye geçilmiştir.

### 3.2 CTU-13 ve CICIDS2018 Bot Verisi Temizliği
Botnet sınıflarında bulunan arka plan DNS (`Port 53`), DHCP, LLMNR ve sıradan TCP akışları (`169.254.169.254`, AWS resolver) filtrelenmiş; yalnızca SPAM, C&C (Komuta Kontrol), Exploit, PortScan ve Malware indirme akışları Bot sınıfında tutulmuştur.

---

## 4. Model Versiyon Kataloğu

| Sürüm | Mimari / Kaynak | FAR (Temiz Ağ) | Recall (Saldırı) | Durum |
| :--- | :--- | :---: | :---: | :--- |
| **v7** | CICIDS2018 Tekil Kaynak | %0.96 | %98.20 (2018 test) | Arşivlendi (`archive/models/ids_model_v7.onnx`) |
| **v8** | CORAL Entegre | %0.04 (Sahte) | **%3.79 (ÇÖKÜŞ)** | Arşivlendi (`archive/models/ids_model_v8_final.pkl`) |
| **v9** | MCFP Kontamine | %99+ | — | Başarısız / Silindi |
| **v10c Baseline** | **Multi-Source Supervised + Temizlenmiş Bot** | **%0.42** | **%83.41 - %92.88** | **ÜRETİM (PRODUCTION)** (`models/baseline/ids_model_v10c_baseline.pkl`) |
| **v10d** | Friday LOIC Eklemeli (Ezber tespit edildi) | %0.21 | Tek tip LOIC ezberi | Arşiv / Test |

---

## 5. Çıkarım Motoru ve Kullanım

### Gerçek Zamanlı Suricata Takibi:
```bash
python3 pipeline/hybrid_inference.py --eve /var/log/suricata/eve.json
```

### Batch Log Analizi:
```bash
python3 pipeline/hybrid_inference.py --eve /path/to/eve.json --batch --output /path/to/alerts.json
```
