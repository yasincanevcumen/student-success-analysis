import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Türkçe görselleştirme için font ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False 

df = pd.read_csv("student-performance-analysis\data\student_study_habits.csv")

def basic_analysis(df):
    """Temel veri analizi"""
    print("="*80)
    print("ÖĞRENCI PERFORMANSI VERİ ANALİZİ")
    print("="*80)
    
    print("\n1. VERİ SETİ GENEL BİLGİLERİ")
    print("-"*50)
    print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Eksik veri: {df.isnull().sum().sum()}")
    
    print("\n2. SAYISAL DEĞİŞKENLER İSTATİSTİKLERİ")
    print("-"*50)
    numeric_cols = df.columns
    stats_df = df[numeric_cols].describe()
    print(stats_df.round(3))
    
    print("\n3. KATEGORİK DEĞİŞKENLER DAĞILIMI")
    print("-"*50)
    
    # Ebeveyn eğitimi
    parent_edu = []
    for _, row in df.iterrows():
        if row['parental_education_High School'] == 1:
            parent_edu.append('Lise')
        elif row['parental_education_Master\'s'] == 1:
            parent_edu.append('Yüksek Lisans')
        elif row['parental_education_PhD'] == 1:
            parent_edu.append('Doktora')
        else:
            parent_edu.append('Diğer')
    
    df['parent_education'] = parent_edu
    print("Ebeveyn Eğitimi Dağılımı:")
    print(df['parent_education'].value_counts())
    
    # Katılım seviyesi
    participation = []
    for _, row in df.iterrows():
        if row['participation_level_Low'] == 1:
            participation.append('Düşük')
        elif row['participation_level_Medium'] == 1:
            participation.append('Orta')
        else:
            participation.append('Yüksek')
    
    df['participation'] = participation
    print("\nKatılım Seviyesi Dağılımı:")
    print(df['participation'].value_counts())
    
    return df

def correlation_analysis(df):
    """Korelasyon analizi"""
    print("\n4. KORELASYON ANALİZİ")
    print("-"*50)
    
    numeric_cols = ['study_hours_per_week', 'sleep_hours_per_day', 
                'attendance_percentage', 'assignments_completed', 'final_grade']
    
    # Korelasyon matrisi
    correlation_matrix = df[numeric_cols].corr()
    
    print("Final Nota Göre Korelasyonlar:")
    final_grade_corr = correlation_matrix['final_grade'].sort_values(ascending=False)
    for var, corr in final_grade_corr.items():
        if var != 'final_grade':
            strength = "Güçlü" if abs(corr) > 0.7 else "Orta" if abs(corr) > 0.3 else "Zayıf"
            direction = "Pozitif" if corr > 0 else "Negatif"
            print(f"{var:25}: {corr:6.3f} ({direction} {strength})")
    
    return correlation_matrix

def visualizations(df, correlation_matrix):
    print("\n5. GÖRSELLEŞTİRMELER OLUŞTURULUYOR...")
    print("-"*50)

    os.makedirs('outputs', exist_ok=True)
    plt.style.use('seaborn-v0_8')

    numeric_cols = ['study_hours_per_week', 'sleep_hours_per_day', 
                    'attendance_percentage', 'assignments_completed', 'final_grade']
    col_labels = ['Çalışma Saati', 'Uyku Saati', 'Devam Oranı', 'Ödev Tamamlama', 'Final Notu']
    
    # 1. Korelasyon Haritası
    plt.figure(figsize=(8,6))
    corr_subset = correlation_matrix.loc[numeric_cols, numeric_cols]
    sns.heatmap(corr_subset, annot=True, cmap='RdYlBu_r', center=0,
                xticklabels=col_labels, yticklabels=col_labels, cbar_kws={'label': 'Korelasyon'})
    plt.title('Korelasyon Haritası', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.show()

    # Çoklu grafikler için 3x3 subplot
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))
    
    # 2. Final Notu Dağılımı
    axs[0, 0].hist(df['final_grade'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axs[0, 0].axvline(df['final_grade'].mean(), color='red', linestyle='--', 
                      label=f'Ortalama: {df["final_grade"].mean():.2f}')
    axs[0, 0].set_xlabel('Final Notu')
    axs[0, 0].set_ylabel('Frekans')
    axs[0, 0].set_title('Final Notu Dağılımı')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 3. Çalışma Saati vs Final Notu
    axs[0, 1].scatter(df['study_hours_per_week'], df['final_grade'], alpha=0.6, color='green')
    z = np.polyfit(df['study_hours_per_week'], df['final_grade'], 1)
    p = np.poly1d(z)
    axs[0, 1].plot(df['study_hours_per_week'], p(df['study_hours_per_week']), "r--", alpha=0.8)
    axs[0, 1].set_xlabel('Haftalık Çalışma Saati')
    axs[0, 1].set_ylabel('Final Notu')
    axs[0, 1].set_title('Çalışma Saati vs Final Notu')
    axs[0, 1].grid(True, alpha=0.3)

    # 4. Uyku Saati vs Final Notu
    axs[0, 2].scatter(df['sleep_hours_per_day'], df['final_grade'], alpha=0.6, color='purple')
    z = np.polyfit(df['sleep_hours_per_day'], df['final_grade'], 1)
    p = np.poly1d(z)
    axs[0, 2].plot(df['sleep_hours_per_day'], p(df['sleep_hours_per_day']), "r--", alpha=0.8)
    axs[0, 2].set_xlabel('Günlük Uyku Saati')
    axs[0, 2].set_ylabel('Final Notu')
    axs[0, 2].set_title('Uyku Saati vs Final Notu')
    axs[0, 2].grid(True, alpha=0.3)

    # 5. Ebeveyn Eğitimi vs Final Notu (seaborn boxplot)
    sns.boxplot(x='parent_education', y='final_grade', data=df, ax=axs[1, 0], palette='Set2')
    axs[1, 0].set_title('Ebeveyn Eğitimi vs Final Notu')
    axs[1, 0].set_xlabel('Ebeveyn Eğitimi')
    axs[1, 0].set_ylabel('Final Notu')

    # 6. Katılım Seviyesi vs Final Notu (seaborn boxplot)
    sns.boxplot(x='participation', y='final_grade', data=df, ax=axs[1, 1], palette='Set3')
    axs[1, 1].set_title('Katılım Seviyesi vs Final Notu')
    axs[1, 1].set_xlabel('Katılım Seviyesi')
    axs[1, 1].set_ylabel('Final Notu')

    # 7. Devam Oranı vs Final Notu
    axs[1, 2].scatter(df['attendance_percentage'], df['final_grade'], alpha=0.6, color='orange')
    z = np.polyfit(df['attendance_percentage'], df['final_grade'], 1)
    p = np.poly1d(z)
    axs[1, 2].plot(df['attendance_percentage'], p(df['attendance_percentage']), "r--", alpha=0.8)
    axs[1, 2].set_xlabel('Devam Oranı')
    axs[1, 2].set_ylabel('Final Notu')
    axs[1, 2].set_title('Devam Oranı vs Final Notu')
    axs[1, 2].grid(True, alpha=0.3)

    # 8. Ödev Tamamlama vs Final Notu
    axs[2, 0].scatter(df['assignments_completed'], df['final_grade'], alpha=0.6, color='brown')
    z = np.polyfit(df['assignments_completed'], df['final_grade'], 1)
    p = np.poly1d(z)
    axs[2, 0].plot(df['assignments_completed'], p(df['assignments_completed']), "r--", alpha=0.8)
    axs[2, 0].set_xlabel('Ödev Tamamlama Oranı')
    axs[2, 0].set_ylabel('Final Notu')
    axs[2, 0].set_title('Ödev Tamamlama vs Final Notu')
    axs[2, 0].grid(True, alpha=0.3)

    # 9. Placeholder subplot (pairplot için)
    axs[2, 1].axis('off')  # Boş bırak

    plt.tight_layout()
    plt.savefig('outputs/multiple_plots.png')
    plt.show()

    # 9. Pair Plot (seaborn, ayrı figür)
    selected_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'final_grade']
    pair_plot = sns.pairplot(df[selected_cols], diag_kind='hist', corner=True)
    pair_plot.fig.suptitle('Çoklu Değişken Analizi (Pairplot)', y=1.02, fontsize=16)
    pair_plot.savefig('outputs/pairplot.png')
    plt.show()

    print("Tüm grafikler 'outputs/' klasörüne kaydedildi.")

def advanced_analysis(df):
    """İleri düzey analiz"""
    print("\n6. İLERİ DÜZEYLİ ANALİZLER")
    print("-"*50)
    
    # Grup analizleri
    print("Ebeveyn Eğitimine Göre Performans:")
    parent_analysis = df.groupby('parent_education')['final_grade'].agg(['mean', 'std', 'count'])
    print(parent_analysis.round(3))
    
    print("\nKatılım Seviyesine Göre Performans:")
    participation_analysis = df.groupby('participation')['final_grade'].agg(['mean', 'std', 'count'])
    print(participation_analysis.round(3))
    
    # İstatistiksel testler
    print("\n7. İSTATİSTİKSEL TESTLER")
    print("-"*50)
    
    # ANOVA testi - Ebeveyn eğitimi grupları arasında fark var mı?
    groups = [group['final_grade'].values for name, group in df.groupby('parent_education')]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"ANOVA Testi (Ebeveyn Eğitimi):")
    print(f"F-istatistik: {f_stat:.3f}, p-değeri: {p_value:.3f}")
    print(f"Sonuç: {'Anlamlı fark var' if p_value < 0.05 else 'Anlamlı fark yok'}")
    
    # Korelasyon testleri
    print("\nKorelasyon Anlamlılık Testleri:")
    numeric_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 'assignments_completed']
    
    for col in numeric_cols:
        corr, p_val = stats.pearsonr(df[col], df['final_grade'])
        print(f"{col:25}: r={corr:6.3f}, p={p_val:.3f} {'*' if p_val < 0.05 else ''}")

def predictive_modeling(df):
    """Tahminsel modelleme"""
    print("\n8. TAHMİNSEL MODELLEME")
    print("-"*50)
    
    # Özellik seçimi
    feature_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 
                    'assignments_completed', 'participation_level_Low', 'participation_level_Medium',
                    'internet_access_Yes', 'parental_education_High School', 
                    'parental_education_Master\'s', 'parental_education_PhD',
                    'extracurricular_Yes', 'part_time_job_Yes']
    
    X = df[feature_cols]
    y = df['final_grade']
    
    # Veri setini böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model 1: Lineer Regresyon
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)
    
    print("Lineer Regresyon Modeli:")
    print(f"R² Skoru: {lr_r2:.3f}")
    print(f"RMSE: {np.sqrt(lr_mse):.3f}")
    
    # Model 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    print("\nRandom Forest Modeli:")
    print(f"R² Skoru: {rf_r2:.3f}")
    print(f"RMSE: {np.sqrt(rf_mse):.3f}")
    
    # Özellik önemlilik analizi
    print("\nÖzellik Önemlilik Sıralaması (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:30}: {row['importance']:.3f}")
    
    return lr_model, rf_model

def generate_insights(df, correlation_matrix):
    """Önemli bulgular ve öneriler"""
    print("\n9. ÖNEMLİ BULGULAR VE ÖNERİLER")
    print("-"*50)
    
    # Korelasyon bulguları
    final_corr = correlation_matrix['final_grade'].drop('final_grade')
    strongest_positive = final_corr.idxmax()
    strongest_negative = final_corr.idxmin()
    
    print("🔍 Önemli Bulgular:")
    print(f"• En güçlü pozitif korelasyon: {strongest_positive} ({final_corr[strongest_positive]:.3f})")
    print(f"• En güçlü negatif korelasyon: {strongest_negative} ({final_corr[strongest_negative]:.3f})")
    
    # Performans grubu analizi
    high_performers = df[df['final_grade'] >= df['final_grade'].quantile(0.75)]
    low_performers = df[df['final_grade'] <= df['final_grade'].quantile(0.25)]
    
    print(f"\n📊 Performans Grubu Karşılaştırması:")
    print(f"Yüksek performans grubu (n={len(high_performers)}):")
    print(f"• Ortalama çalışma saati: {high_performers['study_hours_per_week'].mean():.2f}")
    print(f"• Ortalama uyku saati: {high_performers['sleep_hours_per_day'].mean():.2f}")
    print(f"• Ortalama devam oranı: {high_performers['attendance_percentage'].mean():.2f}")
    print(f"• Ortalama ödev tamamlama: {high_performers['assignments_completed'].mean():.2f}")
    
    print(f"\nDüşük performans grubu (n={len(low_performers)}):")
    print(f"• Ortalama çalışma saati: {low_performers['study_hours_per_week'].mean():.2f}")
    print(f"• Ortalama uyku saati: {low_performers['sleep_hours_per_day'].mean():.2f}")
    print(f"• Ortalama devam oranı: {low_performers['attendance_percentage'].mean():.2f}")
    print(f"• Ortalama ödev tamamlama: {low_performers['assignments_completed'].mean():.2f}")

def generate_conclusion(df, correlation_matrix, feature_importance):
    print("\n" + "="*80)
    print("📌 SONUÇ RAPORU")
    print("="*80)

    final_corr = correlation_matrix['final_grade'].drop('final_grade')
    strongest_pos = final_corr.idxmax()
    strongest_neg = final_corr.idxmin()

    pos_val = final_corr[strongest_pos]
    neg_val = final_corr[strongest_neg]

    pos_strength = "güçlü" if abs(pos_val) > 0.7 else "orta düzeyde" if abs(pos_val) > 0.4 else "zayıf"
    neg_strength = "güçlü" if abs(neg_val) > 0.7 else "orta düzeyde" if abs(neg_val) > 0.4 else "zayıf"

    print(f"""
Bu çalışmada öğrencilerin akademik başarılarını etkileyen faktörler incelenmiştir.

Yapılan korelasyon analizine göre:
• '{strongest_pos}' ile 'final_grade' arasında {pos_strength} pozitif korelasyon (r ≈ {pos_val:.2f}) gözlemlenmiştir.
• '{strongest_neg}' ile 'final_grade' arasında {neg_strength} negatif korelasyon (r ≈ {neg_val:.2f}) saptanmıştır.

Random Forest modeline göre, akademik başarı üzerinde en önemli faktörler:
""")
    for i, row in feature_importance.head(3).iterrows():
        print(f"{i+1}. {row['feature']} ({row['importance']:.3f})")

    print("""
Sonuç olarak:
- Ödev tamamlama oranı, çalışma süresi gibi değişkenler öğrencilerin final notlarını anlamlı şekilde etkilemektedir.
- Öğrencilerin bu alanlarda desteklenmesi, başarılarını artırabilir.

Elde edilen bulgular eğitim politikaları ve öğrenci destek programları için yol gösterici olabilir.
""")
    print("="*80)



if __name__ == "__main__":
    df = basic_analysis(df)
    corr_matrix = correlation_analysis(df)
    visualizations(df, corr_matrix)
    advanced_analysis(df)
    predictive_modeling(df)
    generate_insights(df, corr_matrix)
    lr_model, rf_model = predictive_modeling(df)

    feature_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 
                    'assignments_completed', 'participation_level_Low', 'participation_level_Medium',
                    'internet_access_Yes', 'parental_education_High School', 
                    'parental_education_Master\'s', 'parental_education_PhD',
                    'extracurricular_Yes', 'part_time_job_Yes']
        
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    generate_insights(df, corr_matrix)
    generate_conclusion(df, corr_matrix, feature_importance)
