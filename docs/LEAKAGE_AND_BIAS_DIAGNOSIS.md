# Veri Sızıntısı (Data Leakage) ve Önyargı (Bias) Adli Analiz Raporu

---

## 1. V10d Modelindeki LOIC DDoS Sızıntısı Analizi

### A. Sorunun Ortaya Çıkışı
`trainv10d` modelinde, CICIDS2017 Cuma günkü gerçek LOIC DDoS saldırısından $40.000$ akış eğitime katılmış; model test edildiğinde **tüm eşik değerlerinde ($0.50 \dots 0.90$) %100.00 DDoS Recall** vermiştir. 

Eşik ne kadar yükselirse yükselsin hiç düşmeyen %100.0'lık başarı şüpheli görülmüş ve derinlemesine adli inceleme (forensic audit) başlatılmıştır.

### B. Adli İnceleme Bulguları
1. **Tekil Saldırgan Kaynağı:** $258.326$ akışlık DDoS veri kümesinin **%99.14'ü ($256.095$ akış)** tek bir IP adresinden (`172.16.0.1` $\rightarrow$ `192.168.10.50:80`) gelmektedir.
2. **Rastgele Bölme Hatası (Random Split Leakage):** Eğitim ve test kümeleri zaman bloklarına göre değil rastgele bölündüğü için, aynı mikro-saniyelik LOIC taşkınının ardışık paketleri hem eğitime hem teste dağılmıştır.
3. **Genel DDoS Kavramı Yerine IP/Zamanlama İmzası Ezberi:** Model, genel DDoS davranışını öğrenmek yerine `172.16.0.1` IP'sinin özel TCP bayrak ve bayt dizilimini ezberlemiştir.
4. **Diğer Sınıflara Zararı:** Karar sınırları bu tekil imzaya göre bozulduğu için diğer saldırıların yakalama oranı düşmüştür (DoS recall %87.4 $\rightarrow$ %81.2, Bot recall %87.0 $\rightarrow$ %79.9).

### C. Zamansal Dilim (Temporal Slice) Doğrulaması
Saldırı zaman ekseninde üçe bölünerek test edilmiştir:
* **Başlangıç (İlk 10 Dakika)**
* **Gelişme (Orta 30 Dakika)**
* **Sönümlenme (Son 10 Dakika)**

**Sonuç:** `ids_model_v10c_baseline` modeli bu Cuma saldırısını eğitimde **HİÇ GÖRMEDEN (Zero-Shot)** tamamen genel akış örüntüleriyle **%72.4 ($T=0.84$) ve %97.1 ($T=0.70$)** oranında başarıyla yakalamıştır. Bu nedenle sızıntılı v10d reddedilmiş, **v10c Baseline üretim modeli** olarak kabul edilmiştir.

---

## 2. CTU-13 ve CICIDS2018 Botnet Veri Temizliği (Bias Audit)

### A. CTU-13 Ham Verisindeki Kontaminasyon
* **Sorun:** CTU-13 botnet senaryolarında virüslü makinelerden çıkan tüm akışlar (zararsız DNS sorguları, NTP senkronizasyonları) "Bot" olarak etiketlenmişti ($210.102$ akış).
* **Etki:** Model temiz ağdaki normal DNS sorgularına yanlış Bot alarmı üretiyordu (Bot FAR: %17.0).
* **Çözüm:** Veri kümesi alt etiketlerine göre ayrıştırıldı. Arka plan DNS ve UDP akışları ($89.750$ akış) atıldı; yalnızca saf kötücül akışlar tutuldu:
  - `CC` (Komuta Kontrol trafiği)
  - `Attack / Exploit`
  - `SPAM taşkınları`
  - `Port Tarama (Scan)`
* **Sonuç:** $120.352$ saf kötücül Bot akışı izole edildi.

### B. CICIDS2018 Botnet Verisindeki AWS Resolver Bias'ı
* **Sorun:** CICIDS2018 Botnet pcap'inde bot makinelerin AWS dahili DNS resolver'ına (`172.31.0.2:53`) yaptığı standart sorgular botnet sınıfında yer alıyordu.
* **Çözüm:** Bu dahili resolver sorguları filtrelenerek sadece Ares botnet C&C ve siber saldırı akışları ($37.452$ akış) tutuldu.
* **Sonuç:** Botnet sınıfı FAR'ı %17.0'dan **%0.42'ye** düşürüldü; Recall değeri **%86.38** seviyesinde korundu.
