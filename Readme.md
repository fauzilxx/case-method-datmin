# Catatan Pembagian Kontribusi dan Penggunaan Asistensi AI (Artificial Intelligence)

**Studi Kasus:** Prediksi Harga Tiket Pesawat Menggunakan Algoritma Random Forest Regression    

---

## 1. Pengantar dan Transparansi Akademik
Dokumen ini disusun sebagai catatan refleksi, pemetaan kompetensi, dan transparansi pengerjaan studi kasus. Pengerjaan proyek ini memadukan **pemahaman mandiri mahasiswa** terhadap konsep dasar *data mining* dan metodologi penelitian, serta **pemanfaatan AI (sebagai asisten teknis / *coding partner*)** untuk menyelesaikan modul-modul komputasi yang memiliki kompleksitas matematis atau teknis tinggi.

---

## 2. Pemetaan Bagian Pengerjaan Berdasarkan Tingkat Kesulitan

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TINGKAT MUDAH (EASY)                            │
│  Status: 100% DIKERJAKAN MANDIRI OLEH MAHASISWA                        │
│  - Pemilihan & pemahaman dataset Kaggle (EaseMyTrip India)             │
│  - Identifikasi fitur & pemahaman masalah bisnis (target bimodal)      │
│  - Pembuatan grafik EDA awal (histogram harga, boxplot maskapai)       │
│  - Perumusan Latar Belakang, Rumusan Masalah, dan Tujuan Analisis      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       TINGKAT MENENGAH (MEDIUM)                        │
│  Status: DIKERJAKAN MANDIRI (DENGAN RUJUKAN DOKUMENTASI & TEORI)       │
│  - Pembersihan data standar (penghapusan duplikat, filter price > 0)   │
│  - Rekayasa fitur dasar (One-Hot Encoding kolom kategori)              │
│  - Pembagian data train-test 80:20 terstratifikasi (stratified split)  │
│  - Pelatihan model baseline (Regresi Linear & Decision Tree)           │
│  - Perhitungan metrik evaluasi dasar (RMSE, MAE, R²)                   │
│  - Penyusunan rekomendasi manajerial (The H-15 Rule & harga bisnis)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                        TINGKAT RUMIT (HARD)                            │
│  Status: MEMBUTUHKAN BANTUAN ASISTENSI AI (ASISTEN TEKNIS)             │
│  - Strategi Tuning 2-Tahap pada Data Besar (238k baris) via Sklearn    │
│  - Multi-layer Permutation Importance & Segmented De-composition       │
│  - Formulasi Uji Ketahanan (Ablation Study) & Non-Identifiability      │
│  - Penataan Sintaks LaTeX Laporan, Sitasi IEEE, & Tipografi Kampus     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rincian Pekerjaan Berdasarkan Kategori

### A. Bagian Mudah (Easy) — Dikerjakan Mandiri (100% Mahasiswa)
Bagian ini diselesaikan secara mandiri berdasarkan pemahaman umum konsep data mining:
1. **Pemilihan Studi Kasus & Dataset:** Menemukan dan mengunduh dataset *Flight Price Prediction* dari Kaggle (EaseMyTrip) yang relevan dengan topik *supervised regression*.
2. **Pemahaman Karakteristik Atribut:** Mengidentifikasi nama-nama kolom, tipe data, serta peran masing-masing atribut (kota asal, kota tujuan, maskapai, durasi, dan sisa hari pemesanan).
3. **Pemeriksaan Kesehatan Awal Data:** Menjalankan perintah eksplorasi sederhana seperti `df.info()`, `df.describe()`, dan `df.isna().sum()` untuk memastikan tidak ada sel yang kosong (*zero missing values*).
4. **Visualisasi Eksplorasi Dasar:** Menghasilkan grafik bivariat sederhana menggunakan *Matplotlib* dan *Seaborn*, seperti perbandingan rata-rata harga antar-maskapai dan sebaran jumlah transit (*stops*).
5. **Konseptualisasi Pendahuluan Laporan:** Menyusun latar belakang mengenai volatilitas harga tiket pesawat, pentingnya *dynamic pricing*, serta merumuskan rumusan masalah dan tujuan studi kasus.

---

### B. Bagian Menengah (Medium) — Dikerjakan Mandiri dengan Referensi Dokumentasi
Bagian ini melibatkan penerapan teknis standar *machine learning* menggunakan bantuan dokumentasi pustaka *Scikit-Learn* dan materi perkuliahan:
1. **Pembersihan Data Rutin (`src/01_cleaning.py`):**
   * Menghapus 2.213 baris duplikat penuh menggunakan fungsi `df.drop_duplicates()`.
   * Memvalidasi bahwa seluruh nilai harga tiket bernilai positif (`price > 0`).
   * Melakukan audit pencilan (*outlier*) menggunakan metode IQR standar ($Q1 - 1,5 \times IQR$ hingga $Q3 + 1,5 \times IQR$) dan memutuskan secara sadar untuk mempertahankan harga mahal kelas Bisnis.
2. **Rekayasa Fitur & Encoding (`src/03_preprocessing.py`):**
   * Mengonversi 7 variabel kategorikal menjadi representasi numerik menggunakan `OneHotEncoder(drop='first')` untuk mencegah multikolinearitas (*dummy variable trap*).
   * Menerapkan pembagian data latih (80%) dan data uji (20%) secara terstratifikasi (`stratify=df['class']`) agar rasio kelas Ekonomi dan Bisnis tetap proporsional di kedua himpunan data.
   * Menetapkan keputusan untuk tidak menerapkan *StandardScaler* / *MinMaxScaler* karena algoritma berbasis pohon bersifat *scale-invariant*.
3. **Pelatihan Model Pembanding (*Baseline Modeling* in `src/04_training.py`):**
   * Membangun model pembanding dasar: *Linear Regression* dan *Decision Tree Regressor* tunggal untuk menguji batas performa awal.
   * Menghitung nilai metrik evaluasi standar: $R^2$, RMSE, MAE, dan MAPE pada data pengujian.
4. **Penyusunan Rekomendasi Kasus Bisnis (Bab 4.5):**
   * Menganalisis temuan kurva pemesanan untuk merumuskan aturan praktis bagi konsumen (*The H-15 Rule*).
   * Menarik kesimpulan manajerial bagi maskapai agar tidak membanting harga tiket kelas Bisnis menjelang hari keberangkatan karena sifat permintaannya yang inelastis.

---

### C. Bagian Rumit (Hard) — Membutuhkan Bantuan Asistensi AI
Bagian-bagian ini memiliki tingkat kesulitan matematis, algoritma, atau komputasi yang tinggi, di mana AI digunakan sebagai konsultan teknis untuk membantu mendesain arsitektur solusi:

#### 1. Strategi Tuning 2-Tahap pada Dataset Skala Besar (`src/04_training.py`)
* **Mengapa Rumit?**  
  Dataset latih memiliki ukuran yang sangat masif (**238.352 baris** dengan 30 fitur). Menjalankan `RandomizedSearchCV` atau `GridSearchCV` secara langsung pada seluruh data latih dengan *5-Fold Cross Validation* membutuhkan waktu komputasi belasan jam dan berisiko memicu *Out of Memory (OOM)* pada komputer lokal.
* **Bantuan AI:**  
  AI membantu merancang arsitektur pelatihan **dua tahap efisien**:
  * *Tahap 1:* Melakukan sampling terstratifikasi sebanyak 25.000 baris untuk mengeksekusi `RandomizedSearchCV` (mencari kombinasi terbaik `n_estimators`, `max_features=0.5`, `min_samples_leaf=2`).
  * *Tahap 2:* Memasang parameter terbaik hasil tuning secara otomatis untuk di-*refit* pada keseluruhan 238.352 data latih.
  * Menyusun penanganan data berdimensi tinggi menggunakan format matriks jarang (*Compressed Sparse Row / CSR matrix* via `scipy.sparse`).

#### 2. Interpretasi Tingkat Lanjut & Dekomposisi Per-Segmen (`src/05_interpretation.py`)
* **Mengapa Rumit?**  
  Fitur bawaan Random Forest (`feature_importances_`) hanya menyediakan *Gini Importance* (penurunan impuritas), yang terkenal bias terhadap fitur kardinalitas tinggi dan menutupi kenyataan bahwa pasar tiket terbagi dua. Mahasiswa pada umumnya hanya menampilkan grafik Gini satu lapis.
* **Bantuan AI:**  
  * AI membantu menuliskan logika pengujian *Permutation Feature Importance* pada data uji secara global, yang membuktikan bahwa mengacak kolom `class` menjatuhkan $R^2$ sebesar 163 poin persentase.
  * AI membantu merancang skrip **Permutasi Per-Segmen**: memisahkan data uji menjadi subset Ekonomi dan Bisnis untuk membuktikan bahwa pendorong harga tiket Ekonomi adalah waktu pemesanan (`days_left`, 112 pp), sedangkan Bisnis adalah durasi/rute (`duration`, 71 pp).
  * Membantu menyusun kode sintaks `PartialDependenceDisplay` dari *Scikit-learn* untuk memetakan kurva non-linear berundak (*tebing H-4 dan H-19*).

#### 3. Analisis Identifiabilitas Data (*Ablation Study*)
* **Mengapa Rumit?**  
  Menjelaskan secara teoritis mengapa model Random Forest kolaps menjadi $R^2 = 0,0170$ ketika fitur `class` dihilangkan.
* **Bantuan AI:**  
  AI membantu mengkuantifikasi bahwa **53,6% baris data (159.744 observasi)** memiliki nilai fitur prediktor yang identik persis namun memiliki perbedaan harga tiket hingga 5,7 kali lipat murni karena kelas kursi. AI membantu merumuskan konsep *statistical non-identifiability* dan menghitung batas atas teoritis ($R^2 \le 0,4463$) untuk memperkuat argumentasi akademik.

#### 4. Penataan Sintaks LaTeX Laporan & Penyesuaian Tipografi Akademik
* **Mengapa Rumit?**  
  Menyusun laporan puluhan halaman dalam format LaTeX standar jurnal/kampus sering mengalami kendala teknis seperti tabel melorot ke kanan (*offside/overfull hbox*), inkonsistensi spasi alinea, perataan tengah judul bab otomatis via `titlesec`, serta penulisan persamaan matematika pohon regresi.
* **Bantuan AI:**  
  AI membantu memperbaiki sintaks tabel LaTeX, mengatur format spasi 1.5 via paket `setspace`, mengonversi font ke *Times New Roman* via `newtxtext, newtxmath`, serta menyusun sitasi berstandar IEEE (`[1]` – `[8]`) yang valid dan terindeks Scopus/IEEE.

---

