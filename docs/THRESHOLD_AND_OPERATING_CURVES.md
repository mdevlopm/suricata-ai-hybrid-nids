# Eşik Tarama ve İşletim Eğrileri (ROC / Operating Points)

---

## 1. İşletim Noktası Seçim Kriteri

Yüksek hızlı kurumsal IDS ortamlarında model başarısı tekil bir metrikle (sadece AUC veya sadece Recall) değerlendirilemez. 
İki temel operasyonel kısıt vardır:
1. **Yanlış Alarm Kısıtı (FAR - False Alarm Rate):** SOC analistlerini boğmamak için FAR $\le \%0.50$ olmalıdır.
2. **Saldırı Yakalama Kısıtı (Recall):** Kritik tehditleri kaçırmamak için Recall $\ge \%80.00$ olmalıdır.

---

## 2. Doğrulanmış $192.942$ Temiz Akış ve Saldırı Kümeleri Üzerinde Eşik Taraması

`ids_model_v10c_baseline` modeli üzerinde $T=0.50$'den $T=0.95$'e kadar yapılan detaylı tarama sonuçları:

| Eşik ($T$) | Temiz Ağ FAR (%) | DoS Recall (%) | DDoS Recall (%) | WebAttack (%) | Botnet (%) | Infil (%) | Genel Recall (%) | Operasyonel Durum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50** | %1.85 | %96.40 | %88.50 | %94.20 | %95.10 | %96.00 | **%94.04** | Yüksek Duyarlılık (Test Modu) |
| **0.60** | %1.32 | %94.80 | %85.60 | %92.40 | %93.80 | %94.50 | **%92.22** | Gözlem / Araştırma |
| **0.70** | %0.94 | %92.10 | %81.30 | %89.60 | %91.40 | %92.00 | **%89.28** | Güvenli Ağ Ortamı |
| **0.80** | %0.61 | %89.50 | %76.20 | %86.80 | %88.90 | %89.40 | **%86.16** | Düşük Gürültülü SOC |
| **0.84** | **%0.42** | **%86.04** | **%73.66** | **%83.78** | **%86.38** | **%87.20** | **%83.41** | 👑 **RESMİ ÜRETİM (PRODUCTION)** |
| **0.90** | %0.28 | %78.20 | %64.10 | %74.50 | %79.20 | %81.00 | **%75.40** | Yüksek Güvenilirlik Filtresi |
| **0.95** | %0.12 | %62.40 | %48.50 | %58.20 | %64.80 | %69.10 | **%60.60** | Yalnızca Kesin Alarmlar |

---

## 3. Neden $T = 0.84$ Seçildi?

* **FAR Hedefi:** $192.942$ akışta sadece $805$ yanlış alarm (**%0.42**), belirlenen <%0.50 tavanının altındadır.
* **Recall Gücü:** Saldırıların **%83.41'i** ilk geçişte anında yakalanır.
* **Hibrit Denge:** DoS/DDoS hacimsel saldırıları doğrudan XGBoost ile yakalanıp engellenirken; yavaş ve sinsi saldırılar (Web, Bot) 2. aşamadaki LSTM zaman penceresine ($L=40$) aktarılarak doğrulanır.
