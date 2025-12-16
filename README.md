Markdown

# 👕 Alt & Üst Giyim Sınıflandırma Projesi (Clothing Binary Classification)

Bu proje, Derin Öğrenme (Deep Learning) yöntemleri kullanılarak görüntülerin **"Alt Giyim"** veya **"Üst Giyim"** olarak sınıflandırılmasını amaçlayan bir Makine Öğrenmesi çalışmasıdır. Proje kapsamında üç farklı model mimarisi geliştirilmiş, eğitilmiş ve performansları karşılaştırmalı olarak analiz edilmiştir.

## 🎯 Proje Hakkında

Moda ve e-ticaret alanında görüntülerin otomatik etiketlenmesi büyük önem taşır. Bu proje, temel bir CNN yapısından başlayarak, Transfer Learning (Transfer Öğrenme) ve Özelleştirilmiş Derin Ağlar (Custom Deep CNN) ile model başarısının nasıl artırılabileceğini göstermektedir.

**Temel Hedef:** Verilen bir giysi görselinin pantolon/etek (alt giyim) mi yoksa tişört/gömlek (üst giyim) mi olduğunu yüksek doğrulukla tahmin etmek.

---

## 📂 Depo İçeriği (Repository Structure)

Bu depo, üç farklı yaklaşımı temsil eden Jupyter Notebook dosyalarını içerir:

| Dosya | Açıklama | Mimarisi | Giriş Boyutu |
| :--- | :--- | :--- | :--- |
| **`model1.ipynb`** | Transfer Learning Yaklaşımı | **VGG16** (Önceden Eğitilmiş) | 224x224 px |
| **`model2.ipynb`** | Temel Başlangıç Modeli | **Standart CNN** (Baseline) | 128x128 px |
| **`model3.ipynb`** | Gelişmiş Özelleştirilmiş Model | **Deep CNN + Dropout** | 128x128 px |

---

## 🧠 Model Mimarileri ve Teknik Analiz

Proje kapsamında geliştirilen modellerin teknik detayları ve tasarım kararları aşağıda en ince ayrıntısına kadar açıklanmıştır.

### 1. Model 1: VGG16 Transfer Learning
Bu modelde, ImageNet veri seti üzerinde milyonlarca görüntü ile eğitilmiş **VGG16** mimarisi kullanılmıştır.
* **Neden Seçildi?** Sıfırdan bir model eğitmek yerine, halihazırda kenar, doku ve şekil özelliklerini tanıyan güçlü bir ağın ağırlıklarını kullanmak, özellikle veri seti sınırlıysa daha yüksek başarı sağlar.
* **Teknik Detay:** VGG16 mimarisinin orijinal giriş boyutuna sadık kalmak için görüntüler **224x224** boyutuna yeniden ölçeklendirilmiştir (Rescale).
* **Beklenti:** En yüksek özellik çıkarma (feature extraction) kabiliyeti sayesinde genellikle en stabil sonuçları vermesi beklenir.

### 2. Model 2: Baseline (Temel) CNN
Bu model, projenin referans noktasıdır.
* **Yapı:** Standart Konvolüsyon (Conv2D) ve Havuzlama (MaxPooling) katmanlarından oluşan sığ bir ağdır.
* **Giriş Boyutu:** İşlem maliyetini düşürmek için görüntüler **128x128** piksel olarak işlenir.
* **Amaç:** Hiçbir optimizasyon yapılmadığında modelin ne kadar öğrenebildiğini görmek ve diğer modellerin başarısını kıyaslamak için bir taban (baseline) oluşturmaktır.

### 3. Model 3: Gelişmiş Custom CNN (Fine-Tuned)
Bu model, Model 2'nin üzerine inşa edilmiş ancak aşırı öğrenmeyi (overfitting) engellemek ve başarıyı artırmak için özel olarak optimize edilmiştir.
* **Derinlik:** 4 Bloklu Konvolüsyon yapısı kullanılmıştır (Filtreler: 32 -> 64 -> 128 -> 128). Ağ derinleştikçe model daha soyut özellikleri (kumaş deseni, dikiş yapısı vb.) öğrenebilir.
* **Overfitting Önleme:**
    * `Dropout(0.5)`: Eğitim sırasında nöronların yarısı rastgele kapatılarak modelin ezber yapması engellenmiş, genelleme yapması zorlanmıştır.
* **Optimizasyon:**
    * `Adam(learning_rate=0.0005)`: Standart öğrenme oranı (0.001) yerine daha düşük bir oran seçilmiştir. Bu, modelin minimum hata noktasına (global minimum) daha hassas adımlarla yaklaşmasını sağlar.
* **Çıkış Katmanı:** İkili sınıflandırma (Alt/Üst) yapıldığı için çıkışta 2 nöron ve `softmax` (veya binary duruma göre sigmoid) aktivasyonu kullanılmıştır.

---

## 📊 Veri Seti ve Hazırlık

Veri seti Google Drive üzerinden çekilmektedir. Kodlar çalıştırılmadan önce veri setinin aşağıdaki yapıda olduğundan emin olunmalıdır:

```text
/content/drive/My Drive/makine_ogrenmesi_veriseti
    ├── training
    │   ├── alt_giyim  (Label 0)
    │   └── ust_giyim  (Label 1)
    └── validation
        ├── alt_giyim
        └── ust_giyim
Rescale: Tüm görüntüler 1./255 ile normalize edilerek piksel değerleri 0-1 arasına çekilmiştir.

Batch Size: 32 (Her iterasyonda 32 görüntü işlenir).

🛠️ Kurulum ve Çalıştırma
Bu projeyi Google Colab üzerinde çalıştırmak için aşağıdaki adımları izleyin:

Bu depoyu klonlayın veya .ipynb dosyalarını indirin.

Google Colab'i açın ve dosyaları yükleyin.

Google Drive bağlantısını sağlayın:

Python

from google.colab import drive
drive.mount('/content/drive')
base_dir değişkenini kendi veri seti yolunuzla güncelleyin.

Hücreleri sırasıyla çalıştırın.

Gerekli Kütüphaneler
Python

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
📈 Sonuçların Değerlendirilmesi
Her modelin eğitimi bittiğinde grafik_ciz fonksiyonu ile Accuracy (Doğruluk) ve Loss (Kayıp) grafikleri çizdirilir.

İyi bir modelde: "Eğitim" ve "Test" (Validation) çizgileri birbirine yakın ve yukarı doğru (Accuracy için) hareket etmelidir.

Overfitting (Aşırı Öğrenme): Eğitim başarısı %99 iken Test başarısı %70'lerde kalıyorsa model ezberlemiş demektir (Model 3'teki Dropout bunu engellemek içindir).

Bu proje Isparta Uygulamalı Bilimler Üniversitesi Bilgisayar Mühendisliği Makine Öğrenmesi dersi kapsamında hazırlanmıştır.
