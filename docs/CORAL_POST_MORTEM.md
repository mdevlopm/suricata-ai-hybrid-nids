# CORAL (Correlation Alignment) Post-Mortem & Fail-Safe Raporu

---

## 1. Yönetici Özeti (Executive Summary)

Aylarca süren denemelerde, sentetik eğitim verisi (CICIDS2018) ile gerçek dünya Suricata trafiği (CICIDS2017) arasındaki alan kaymasını (domain shift) gidermek amacıyla **CORAL (Domain Adaptation via Covariance Alignment)** algoritması kullanılmıştır. 

CORAL, hedef alandaki Yanlış Alarm Oranını (FAR) **%74.68'den %0.04'e** düşürerek ilk etapta başarılı görünmüş; ancak uçtan uca yapılan doğrulama testlerinde **gerçek saldırıları yakalama oranını (Recall) %98.57'den %3.79'a indirdiği (yani saldırıların %96.21'ini kaçırdığı)** tespit edilmiştir.

Bu belge, CORAL'ın matematiksel çöküş mekanizmasını, modelde nasıl bir "sessiz körlük" yarattığını ve neden kalıcı olarak terk edildiğini belgeler.

---

## 2. CORAL'ın Çalışma Prensibi

CORAL, kaynak (eğitim) dağılımının kovaryans matrisi ($C_S$) ile hedef (canlı ağ) dağılımının kovaryans matrisi ($C_T$) arasındaki ikinci derece istatistiksel mesafeyi minimize etmeyi hedefler:

$$\min_{A} \| C_{\hat{S}} - C_T \|_F^2 = \min_{A} \| A^T C_S A - C_T \|_F^2$$

Dönüşüm matrisi $A$, hedef kovaryansın beyazlatılması (whitening) ve renklendirilmesi (re-coloring) ile elde edilir:

$$x_{\text{coral}} = x \cdot C_T^{-1/2} \cdot C_S^{1/2}$$

---

## 3. Matematiksel Çöküşün Nedeni (Neden İflas Etti?)

1. **Hedef Dağılımın Sadece Benign (Temiz) Olması:**
   Canlı ağdan toplanan hedef dağılım $C_T$, neredeyse tamamen temiz ofis trafiğinden (Benign) oluşmaktadır. Bu matriste hiçbir saldırı varyansı bulunmaz.

2. **Saldırı Vektörlerinin Temiz Uzaya Döndürülmesi (Subspace Projection):**
   Hedef kovaryans matrisinin ters karekökü ($C_T^{-1/2}$) ile çarpma işlemi, ağa gelen her türlü çok-değişkenli anomali ve saldırı örüntüsünü (DDoS, Brute Force, Web Attack) temiz ofis trafiğinin varyans eksenlerine zorla izdüşürür.

3. **Görselleştirilmiş Çöküş:**
   ```
   [Gerçek Saldırı Vektörü] (Yüksek varyans, belirgin saldırı izi)
                │
                ▼ (CORAL Kovaryans Beyazlatması)
   [Dönüştürülmüş Vektör] (Temiz ofis trafiğine matematiksel olarak benzetilmiş)
                │
                ▼
   [XGBoost Karar Sınırı] ──> "Bu bir normal Benign trafiğidir" (Recall = %3.79)
   ```

---

## 4. Sayısal Kanıtlar ve Karşılaştırma Tablosu

| Metrik | Ham Model (No CORAL) | CORAL Uygulanmış Model | Değişim / Etki |
| :--- | :---: | :---: | :---: |
| **Benign Yanlış Alarm (FAR)** | %0.42 | %0.04 | Görünüşte düştü (Yanılsama) |
| **DoS Recall** | **%86.04** | **%1.20** | 🔴 %84.84 Saldırı Kaçırma |
| **DDoS Recall** | **%73.66** | **%0.85** | 🔴 %72.81 Saldırı Kaçırma |
| **Web Attack Recall** | **%83.78** | **%4.10** | 🔴 %79.68 Saldırı Kaçırma |
| **Botnet Recall** | **%86.38** | **%8.12** | 🔴 %78.26 Saldırı Kaçırma |
| **Infiltration Recall** | **%87.20** | **%4.68** | 🔴 %82.52 Saldırı Kaçırma |
| **GENEL SALDIRI RECALL** | **%83.41** | **%3.79** | 🔴 **%79.62 ÇÖKÜŞ** |

---

## 5. Alınan Karar ve Çözüm Yolu

1. **CORAL Kalıcı Olarak Kaldırıldı:** `hybrid_inference.py` ve tüm eğitim scriptlerinden CORAL bağımlılıkları temizlendi.
2. **Çok Kaynaklı Süpervize Eğitim (Multi-Source Supervised):** Matematiksel post-hoc dönüşümler yerine, modele CTU-13 ve CICIDS2017'den temizlenmiş gerçek Suricata-native saldırı örnekleri doğrudan gösterildi (`ids_model_v10c_baseline.pkl`).
3. **Kural:** Gelecekte hiçbir şart altında CORAL, OT (Optimal Transport) veya benzeri hedefsiz kovaryans beyazlatma yöntemleri sisteme eklenemez.
