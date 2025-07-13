# 🎓 Öğrenci Akademik Başarısı Üzerine Veri Temelli Bir Analiz

## 🔍 Proje Amacı

Bu proje, öğrencilerin akademik başarılarını etkileyen temel faktörleri istatistiksel ve makine öğrenmesi temelli yaklaşımlarla incelemeyi amaçlamaktadır. Python programlama dili kullanılarak yürütülen bu analizde, öğrencilerin çalışma alışkanlıkları, uyku düzeni, ebeveynlerinin eğitim düzeyi ve ödev tamamlama oranı gibi değişkenlerin final notu üzerindeki etkileri araştırılmıştır.

---

## 📂 Veri Seti Özellikleri

Çalışmada kullanılan veri seti, her biri bir öğrenciye ait olmak üzere 500 gözlem içermektedir. Ana değişkenler şunlardır:

- `study_hours_per_week`: Haftalık çalışma süresi (saat)
- `sleep_hours_per_day`: Günlük uyku süresi (saat)
- `attendance_percentage`: Derse devam oranı (%)
- `assignments_completed`: Ödev tamamlama oranı (%)
- `final_grade`: Final sınavından alınan not
- `parental_education_*`: Ebeveyn eğitim düzeyi (Lise, Yüksek Lisans, Doktora vb.)
- `participation_level_*`: Derse katılım düzeyi (Düşük, Orta, Yüksek)

Veri setindeki kategorik değişkenler, daha kolay yorumlanabilmesi amacıyla dönüştürülmüştür.

---

## 🧪 Kullanılan Yöntemler

Proje kapsamında şu analiz teknikleri uygulanmıştır:

- Temel betimleyici istatistikler
- Korelasyon analizi (Pearson)
- ANOVA testi (ebeveyn eğitimi grupları arası fark için)
- Görselleştirmeler (heatmap, scatter plot, boxplot)
- Regresyon modelleri:
  - Lineer regresyon
  - Random Forest regresyon

---

## 📈 Bulgular

- **En güçlü pozitif korelasyon**, ödev tamamlama oranı ile final notu arasında bulunmuştur (**r = 0.64**).
- Haftalık çalışma süresi ile başarı arasında **orta düzeyde pozitif** bir ilişki gözlenmiştir (**r = 0.57**).
- Devam oranı ile final notu arasında **negatif ve zayıf** bir korelasyon saptanmıştır (**r = -0.21**).
- ANOVA sonucuna göre ebeveyn eğitimi ile final notu arasında **istatistiksel olarak anlamlı bir fark gözlenmemiştir** (p > 0.05).
- Random Forest modelinde, **ödev tamamlama oranı ve çalışma süresi**, başarıyı en iyi tahmin eden değişkenler olarak öne çıkmıştır.

---

## 💻 Kurulum ve Kullanım

### Gereksinimler

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

### Kurulum

```bash
git clone https://github.com/ycevcumen/student-performance-analysis.git
cd student-performance-analysis
pip install -r requirements.txt
