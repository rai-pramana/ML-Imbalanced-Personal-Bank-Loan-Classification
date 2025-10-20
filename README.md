# ML-Imbalanced-Personal-Bank-Loan-Classification

Proyek Machine Learning untuk prediksi pendapatan dan klasifikasi calon nasabah pinjaman potensial bank menggunakan teknik regresi linier dan algoritma Decision Tree dan Random Forest pada dataset yang tidak seimbang.

## 📋 Deskripsi Proyek

Proyek ini terdiri dari dua analisis utama:

1. **Prediksi Pendapatan Nasabah Bank dengan Regresi Linier**

    - Memprediksi pendapatan tahunan nasabah untuk meningkatkan personalisasi layanan
    - Mengoptimalkan strategi pemasaran dan pengelolaan risiko kredit
    - Meningkatkan efisiensi operasional bank

2. **Klasifikasi Calon Nasabah Pinjaman Potensial dengan Decision Tree dan Random Forest**
    - Mengidentifikasi nasabah yang kemungkinan besar akan membeli pinjaman pribadi
    - Meningkatkan efisiensi pemrosesan aplikasi pinjaman
    - Mengoptimalkan alokasi sumber daya untuk pemasaran produk pinjaman

## 👥 Tim Pengembang

**Kelompok 7 - Machine Learning (C)**

-   I Kadek Rai Pramana (2105551094)
-   Gusti Ngurah Bagus Picesa Kresna Mandala (2105551097)

## 📊 Dataset

Dataset berisi informasi 5000 nasabah bank dengan 14 variabel:

### Variabel Numerik

-   **Age**: Usia nasabah (tahun)
-   **Experience**: Pengalaman kerja (tahun)
-   **Income**: Pendapatan tahunan (ribuan)
-   **CCAvg**: Rata-rata pengeluaran kartu kredit per bulan (ribuan)
-   **Mortgage**: Nilai hipotek (ribuan)
-   **ZIP Code**: Kode pos tempat tinggal

### Variabel Kategorikal

-   **Family**: Jumlah anggota keluarga
-   **Education**: Tingkat pendidikan (1=Sarjana, 2=Master, 3=Profesional)
-   **Personal Loan**: Status pinjaman pribadi (Target untuk klasifikasi)
-   **Securities Account**: Memiliki akun sekuritas (0/1)
-   **CD Account**: Memiliki akun deposito berjangka (0/1)
-   **Online**: Menggunakan layanan perbankan online (0/1)
-   **CreditCard**: Memiliki kartu kredit (0/1)

## 🛠️ Teknologi dan Library

```python
# Data Processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

# Statistical Analysis
from scipy.stats import zscore, ttest_ind, chi2_contingency, levene, f_oneway
from pingouin import welch_anova
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, recall_score, precision_score, f1_score,
                            roc_auc_score, classification_report, RocCurveDisplay,
                            ConfusionMatrixDisplay)
```

## 🔄 Alur Kerja Proyek

### 1. Pembersihan Data (Data Cleaning)

-   ✅ Tidak ada missing values
-   ✅ Tidak ada data duplikat
-   🔧 Penanganan noise pada variabel `zip_code` (1 data dengan 4 digit)
-   🔧 Penanganan nilai negatif pada `experience` (konversi ke nilai absolut)
-   🔧 Penghapusan outlier pada `mortgage` menggunakan Z-score (threshold > 3)
-   🔧 Transformasi `cc_avg` dari bulanan ke tahunan

### 2. Analisis Eksploratori (EDA)

#### Analisis Univariat

-   Distribusi variabel kategorikal dan numerik
-   Visualisasi menggunakan countplot, histogram, boxplot, dan KDE

#### Analisis Bivariat

-   Hubungan variabel independen dengan target (`income` dan `personal_loan`)
-   Scatter plot dan stacked bar plot untuk visualisasi

#### Analisis Korelasi

-   Korelasi Spearman antar variabel
-   Penghapusan variabel `experience` (korelasi tinggi dengan `age`: ρ = 0.99)

### 3. Pengujian Hipotesis

#### t-test (Variabel Numerik)

-   Menguji perbedaan rata-rata antara kelompok pinjaman

#### Chi-square (Variabel Kategorikal)

-   Menguji hubungan antara variabel kategorikal dan status pinjaman

#### ANOVA (Welch)

-   Menguji perbedaan rata-rata di antara kelompok kategorikal

## 📈 Hasil Analisis

### Prediksi Pendapatan (Linear Regression)

**Model Terpilih**: Regresi Linier Berganda

-   **Variabel**: `cc_avg` dan `personal_loan`
-   **Persamaan**: `income = 40.88 + (46.23 × cc_avg) + (1.18 × personal_loan)`

**Performa Model**:

-   R² Score: **0.511** (51.1% variasi dijelaskan)
-   Mean Squared Error: **988.72**
-   Mean Absolute Error: **24.85**

**Interpretasi**: Model dapat memprediksi pendapatan dengan cukup baik, namun masih ada ruang untuk perbaikan dengan penambahan variabel prediktor lain.

### Klasifikasi Pinjaman (Decision Tree & Random Forest)

#### Penanganan Imbalanced Dataset

-   Dataset tidak seimbang: **91.2%** kelas 0, **8.8%** kelas 1
-   Strategi: Stratified sampling dan class weighting

#### Hasil Perbandingan Model

| Model             | Accuracy   | Precision  | Recall     | F1-Score   | AUC        |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest** | **98.88%** | **95.24%** | **91.95%** | **93.57%** | **99.87%** |
| Decision Tree     | 98.37%     | 92.77%     | 88.51%     | 90.59%     | 98.52%     |

**Model Terbaik**: **Random Forest**

-   Kesalahan prediksi: **11 dari 979** nasabah (1.12%)
-   Fitur paling penting: `education`, `income`, `family`, `cc_avg`, `cd_account`

#### Hyperparameter Optimal (Random Forest)

```python
{
    'n_estimators': 50,
    'max_depth': 11,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': {0: 0.58, 1: 0.42}
}
```

## 🚀 Cara Menggunakan

### 1. Clone Repository

```bash
git clone https://github.com/rai-pramana/ML-Imbalanced-Personal-Bank-Loan-Classification.git
cd ML-Imbalanced-Personal-Bank-Loan-Classification
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy pingouin statsmodels scikit-learn ydata-profiling gdown
```

### 3. Jalankan Notebook

```bash
jupyter notebook "Kode Program_UAS_ML (C)_Kelompok 7.ipynb"
```

## 📊 Struktur File

```
ML-Imbalanced-Personal-Bank-Loan-Classification/
│
├── Kode Program_UAS_ML (C)_Kelompok 7.ipynb  # Main notebook
├── README.md                                   # Dokumentasi proyek
├── bank_personal_loan_modelling.xlsx          # Dataset
├── dt_result.csv                              # Hasil Decision Tree
└── rf_result.csv                              # Hasil Random Forest
```

## 🎯 Kesimpulan Utama

### Prediksi Pendapatan

-   Model regresi linier dapat menjelaskan 51.1% variasi pendapatan
-   Variabel `cc_avg` memiliki pengaruh paling besar terhadap pendapatan
-   Diperlukan variabel tambahan untuk meningkatkan akurasi prediksi

### Klasifikasi Pinjaman

-   **Random Forest mengungguli Decision Tree** dalam semua metrik evaluasi
-   F1-score Random Forest: **93.57%** vs Decision Tree: **90.59%**
-   Tingkat akurasi sangat tinggi: **98.88%**
-   Model berhasil menangani dataset yang tidak seimbang dengan efektif

### Fitur Penting

1. **Education** (Tingkat pendidikan)
2. **Income** (Pendapatan)
3. **Family** (Jumlah anggota keluarga)
4. **CC_Avg** (Pengeluaran kartu kredit)
5. **CD_Account** (Akun deposito berjangka)

## 💡 Rekomendasi

### Untuk Bank

1. **Fokus pada nasabah dengan pendidikan tinggi** untuk kampanye pinjaman pribadi
2. **Target nasabah dengan pendapatan dan pengeluaran kartu kredit tinggi**
3. **Prioritaskan nasabah yang sudah memiliki CD account**
4. Gunakan model Random Forest untuk **screening otomatis** calon peminjam potensial

### Pengembangan Model

1. Tambahkan variabel prediktor baru untuk meningkatkan R² regresi
2. Eksplorasi teknik ensemble learning lainnya (XGBoost, LightGBM)
3. Implementasi teknik oversampling/undersampling (SMOTE, ADASYN)
4. Lakukan feature engineering untuk meningkatkan performa model

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademis - Ujian Akhir Semester Machine Learning (C).

## 📧 Kontak

Untuk pertanyaan atau diskusi lebih lanjut, silakan hubungi tim pengembang melalui repository issues.

---

**Catatan**: Dataset yang digunakan dalam proyek ini adalah data simulasi untuk tujuan pembelajaran dan tidak mewakili data nasabah bank yang sebenarnya.
