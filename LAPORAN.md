# LAPORAN MID PROJECT
## REKAYASA KOMPUTASIONAL
### "OPTIMASI PEMBOBOTAN FITUR TF-IDF PADA ANALISIS SENTIMEN ULASAN RESTORAN BERBAHASA INDONESIA MENGGUNAKAN PARTICLE SWARM OPTIMIZATION (PSO)"

Oleh:
Muhammad Naim
(105841108124)

PRODI INFORMATIKA
FAKULTAS TEKNIK
UNIVERSITAS MUHAMMADIYAH MAKASSAR
2026

---

## KATA PENGANTAR

Puji syukur ke hadirat Allah SWT karena atas rahmat dan karunia-Nya sehingga penulis dapat menyelesaikan laporan Mid Project mata kuliah Rekayasa Komputasional yang berjudul Optimasi Pembobotan Fitur TF-IDF pada Analisis Sentimen Ulasan Restoran Berbahasa Indonesia Menggunakan Particle Swarm Optimization (PSO) dengan baik. Laporan ini disusun sebagai salah satu bentuk pemenuhan tugas akademik sekaligus penerapan metode komputasi cerdas dalam bidang pengolahan bahasa alami (Natural Language Processing).

Dalam laporan ini, penulis membahas perancangan dan implementasi sistem analisis sentimen ulasan restoran menggunakan kombinasi TF-IDF (Term Frequency-Inverse Document Frequency) dan Logistic Regression, yang kemudian dioptimasi menggunakan algoritma Particle Swarm Optimization (PSO) untuk menentukan bobot fitur yang paling optimal. Sistem yang dikembangkan diharapkan dapat meningkatkan akurasi klasifikasi sentimen (positif/negatif) dibandingkan model dasar (baseline) tanpa optimasi.

Penulis menyadari bahwa laporan ini masih memiliki berbagai keterbatasan dan kekurangan. Oleh karena itu, kritik dan saran yang membangun sangat diharapkan untuk perbaikan di masa mendatang. Penulis juga mengucapkan terima kasih kepada dosen pengampu serta semua pihak yang telah memberikan dukungan dalam penyusunan laporan ini. Semoga laporan ini dapat memberikan manfaat bagi pembaca dan menjadi referensi dalam pengembangan sistem rekayasa komputasional berbasis metaheuristik.

Makassar, 30 Mei 2026

Muhammad Naim

---

## DAFTAR ISI

- KATA PENGANTAR
- DAFTAR ISI
- BAB I PENDAHULUAN
  - 1.1 Latar Belakang
- BAB II HIPOTESIS
  - 2.1 Hipotesis Umum
  - 2.2 Hipotesis Khusus
    - 2.2.1 Hipotesis Pembobotan Fitur PSO
    - 2.2.2 Hipotesis Pelabelan Berbasis Leksikon
    - 2.2.3 Hipotesis Acceptance Gate PSO
    - 2.2.4 Hipotesis Evaluasi
- BAB III RANCANGAN SISTEM
  - 3.1 Arsitektur Sistem
  - 3.2 Parameter Konfigurasi Sistem
  - 3.3 Kelas Klasifikasi
- BAB IV IMPLEMENTASI
  - 4.1 Pipeline Preprocessing Data
    - 4.1.1 Pembersihan Teks (Text Cleaning)
    - 4.1.2 Stopword Removal dan Stemming
    - 4.1.3 Pelabelan Sentimen Berbasis Leksikon
  - 4.2 Ekstraksi Fitur dan Proses Training
    - 4.2.1 Ekstraksi Fitur TF-IDF
    - 4.2.2 Model Baseline (Logistic Regression)
    - 4.2.3 Fungsi Fitness PSO
    - 4.2.4 Algoritma Particle Swarm Optimization
  - 4.3 Evaluasi Model
    - 4.3.1 Metrik Evaluasi
    - 4.3.2 Confusion Matrix
    - 4.3.3 Skema Repeated Stratified K-Fold
  - 4.4 Acceptance Gate dan Prediksi Hybrid
  - 4.5 Pengujian Black Box
- BAB V PENUTUP
  - 5.1 Kesimpulan
  - 5.2 Saran

---

## BAB I PENDAHULUAN

### 1.1 Latar Belakang

Ulasan (review) pelanggan pada platform digital merupakan salah satu sumber data yang sangat berharga bagi pelaku usaha, khususnya di bidang restoran, untuk memahami kepuasan pelanggan terhadap layanan dan produk yang diberikan. Namun, jumlah ulasan yang terus bertambah membuat proses analisis secara manual menjadi tidak efisien dan rentan terhadap bias subjektif. Oleh karena itu, diperlukan sebuah sistem otomatis yang mampu mengklasifikasikan sentimen ulasan (positif atau negatif) secara cepat dan akurat.

Salah satu pendekatan populer dalam analisis sentimen berbasis teks adalah representasi fitur menggunakan TF-IDF yang dikombinasikan dengan model klasifikasi seperti Logistic Regression. Meskipun pendekatan ini cukup efektif, performa model sangat bergantung pada seberapa relevan bobot setiap fitur (kata) terhadap tugas klasifikasi. Fitur dengan bobot TF-IDF standar tidak selalu mencerminkan tingkat kepentingan kata tersebut secara optimal untuk membedakan sentimen positif dan negatif.

Untuk mengatasi permasalahan tersebut, laporan ini mengusulkan penerapan algoritma metaheuristik Particle Swarm Optimization (PSO) untuk mengoptimasi bobot fitur TF-IDF sebelum digunakan sebagai input model Logistic Regression. Proses pelabelan data pada sistem ini dilakukan secara lexicon-based (berbasis kamus kata positif dan negatif Bahasa Indonesia), bukan berdasarkan rating bintang, sehingga label yang dihasilkan murni berdasarkan konten teks ulasan. Sistem ini juga dilengkapi dengan mekanisme acceptance gate yang memastikan bobot hasil PSO hanya digunakan apabila benar-benar memberikan peningkatan performa dibandingkan model baseline, serta evaluasi menggunakan skema Repeated Stratified K-Fold agar hasil lebih dapat dipercaya secara statistik.

---

## BAB II HIPOTESIS

### 2.1 Hipotesis Umum

Berdasarkan kajian literatur dan analisis permasalahan, diajukan hipotesis umum sebagai berikut:

"Pembobotan fitur TF-IDF yang dioptimasi menggunakan algoritma Particle Swarm Optimization (PSO) mampu meningkatkan performa klasifikasi sentimen ulasan restoran berbahasa Indonesia dibandingkan model Logistic Regression baseline tanpa optimasi bobot fitur."

### 2.2 Hipotesis Khusus

#### 2.2.1 Hipotesis Pembobotan Fitur PSO

Algoritma PSO dengan mekanisme inertia weight yang menurun secara linear dan strategi reinisialisasi partikel saat stagnasi akan mampu menemukan kombinasi bobot fitur yang menghasilkan nilai F1-score macro lebih tinggi dibandingkan bobot fitur seragam (netral).

#### 2.2.2 Hipotesis Pelabelan Berbasis Leksikon

Pelabelan sentimen berbasis kamus kata positif dan negatif dengan mempertimbangkan kata negasi (negation handling) dalam jendela kata tertentu akan menghasilkan label yang lebih merepresentasikan konten teks ulasan dibandingkan pelabelan berbasis rating bintang semata.

#### 2.2.3 Hipotesis Acceptance Gate PSO

Penerapan mekanisme acceptance gate, yang membandingkan fitness bobot hasil PSO dengan fitness bobot netral pada validasi silang internal (inner cross-validation), akan mencegah penggunaan bobot PSO yang tidak signifikan lebih baik, sehingga model tetap stabil dan tidak overfitting terhadap data latih.

#### 2.2.4 Hipotesis Evaluasi

Evaluasi menggunakan skema Repeated Stratified K-Fold akan menghasilkan estimasi performa model (accuracy dan F1-score) yang lebih stabil dan memiliki varians lebih rendah dibandingkan evaluasi dengan satu kali pembagian data train-test (single 80/20 split).

---

## BAB III RANCANGAN SISTEM

### 3.1 Arsitektur Sistem

Sistem analisis sentimen ulasan restoran yang dibangun terdiri dari beberapa tahapan utama yang saling terintegrasi dalam satu alur (pipeline), yaitu: (1) Pemuatan dan Pembersihan Dataset, (2) Preprocessing Teks Bahasa Indonesia, (3) Pelabelan Sentimen Berbasis Leksikon, (4) Ekstraksi Fitur TF-IDF, (5) Optimasi Bobot Fitur dengan PSO, (6) Pelatihan dan Evaluasi Model, serta (7) Visualisasi dan Analisis Hasil. Berikut adalah gambaran komponen utama sistem:

| Komponen | Modul/Fungsi | Fungsi Utama |
|---|---|---|
| Pemuatan Dataset | `read_csv_robust()` | Membaca file `dataset_ulasan_restoran.csv` dengan deteksi encoding otomatis |
| Deteksi Kolom | `find_col()` | Mendeteksi kolom teks ulasan dan rating secara otomatis |
| Preprocessing | `clean_text()`, `preprocess_for_tfidf()` | Pembersihan teks, stopword removal, stemming Bahasa Indonesia |
| Pelabelan | `sentiment_score()`, `score_to_label()` | Penghitungan skor sentimen berbasis leksikon dan penentuan label |
| Ekstraksi Fitur | `TfidfVectorizer` | Konversi teks menjadi representasi vektor TF-IDF |
| Optimasi | `Particle`, `pso_optimize()`, `make_fitness_function()` | Optimasi bobot fitur menggunakan algoritma PSO |
| Model Klasifikasi | `LogisticRegression` | Klasifikasi sentimen positif/negatif |
| Evaluasi | `run_one_fold()`, `RepeatedStratifiedKFold` | Perhitungan metrik dan validasi silang |
| Prediksi Hybrid | `hybrid_predict()` | Penggabungan prediksi PSO dan baseline berdasarkan margin probabilitas |
| Visualisasi | `matplotlib.pyplot` | Grafik metrik, confusion matrix, kurva konvergensi PSO |

### 3.2 Parameter Konfigurasi Sistem

Seluruh parameter utama sistem dikonfigurasi secara terpusat pada bagian awal `main.py` untuk memudahkan penyesuaian:

| Parameter | Nilai | Keterangan |
|---|---|---|
| MAX_TFIDF_FEATURES | 150 | Jumlah maksimum fitur (kata) TF-IDF yang digunakan |
| N_OUTER_SPLITS | 5 | Jumlah fold pada outer Repeated Stratified K-Fold |
| N_OUTER_REPEATS | 3 | Jumlah pengulangan skema outer K-Fold |
| N_INNER_SPLITS | 3 | Jumlah fold pada inner cross-validation (fitness PSO) |
| N_INNER_REPEATS | 2 | Jumlah pengulangan skema inner K-Fold |
| REG_LAMBDA | 0.08 | Koefisien regularisasi fitness terhadap penyimpangan bobot dari 1.0 |
| N_PARTICLES | 20 (8 saat QUICK_TEST) | Jumlah partikel dalam swarm PSO |
| N_ITERATIONS | 30 (10 saat QUICK_TEST) | Jumlah iterasi maksimum PSO |
| C1, C2 | 1.5, 1.5 | Koefisien kognitif dan sosial PSO |
| W_MIN, W_MAX | 0.4, 0.9 | Rentang inertia weight PSO |
| PSO_MIN_IMPROVEMENT | 0.01 | Ambang minimum perbaikan fitness agar bobot PSO diterima |
| PSO_TIE_MARGIN | 0.03 | Ambang margin probabilitas untuk fallback ke prediksi baseline |
| RANDOM_STATE | 42 | Seed acak untuk memastikan hasil dapat direproduksi |

### 3.3 Kelas Klasifikasi

Sistem mengklasifikasikan ulasan restoran ke dalam dua kelas sentimen berikut, yang ditentukan melalui pendekatan lexicon-based (bukan dari rating bintang):

| Kode Label | Nama Kelas | Deskripsi |
|---|---|---|
| positive | Positif | Skor sentimen leksikon bernilai lebih besar dari nol (dominan kata bermakna positif) |
| negative | Negatif | Skor sentimen leksikon bernilai lebih kecil dari nol (dominan kata bermakna negatif) |
| ambiguous | Ambigu (dibuang) | Skor sentimen bernilai nol; data dikeluarkan dari proses pemodelan |

---

## BAB IV IMPLEMENTASI

### 4.1 Pipeline Preprocessing Data

#### 4.1.1 Pembersihan Teks (Text Cleaning)

Fungsi `clean_text()` pada `main.py` digunakan untuk membersihkan teks ulasan mentah sebelum diproses lebih lanjut:
1. Mengubah seluruh teks menjadi huruf kecil (lowercase)
2. Menghapus URL (pola `http\S+` dan `www\S+`)
3. Menghapus karakter selain huruf dan spasi menggunakan regex
4. Menormalkan spasi berlebih menjadi satu spasi
5. Mengurangi pengulangan karakter berturut-turut (misalnya "enaaak" menjadi "enak")

#### 4.1.2 Stopword Removal dan Stemming

Fungsi `preprocess_for_tfidf()` melanjutkan hasil `clean_text()` dengan:
1. Menghapus stopword Bahasa Indonesia menggunakan `StopWordRemoverFactory` dari pustaka Sastrawi
2. Melakukan stemming (pengubahan kata berimbuhan menjadi kata dasar) menggunakan `StemmerFactory` dari pustaka Sastrawi

Hasil dari tahap ini disimpan pada kolom `clean_text` yang digunakan sebagai input bagi `TfidfVectorizer`.

#### 4.1.3 Pelabelan Sentimen Berbasis Leksikon

Pelabelan dilakukan tanpa menggunakan rating bintang, melainkan berdasarkan skor kata pada `POSITIVE_LEXICON` dan `NEGATIVE_LEXICON`:

| Komponen | Keterangan |
|---|---|
| POSITIVE_LEXICON | Kamus kata positif Bahasa Indonesia dengan bobot 1-2 (mis. "enak", "puas", "ramah") |
| NEGATIVE_LEXICON | Kamus kata negatif Bahasa Indonesia dengan bobot 1-2 (mis. "kecewa", "buruk", "kotor") |
| NEGATION_WORDS | Kata negasi (mis. "tidak", "bukan", "kurang") yang membalik polaritas skor |
| NEGATION_WINDOW | Jendela 2 kata sebelum kata sentimen untuk mendeteksi negasi |

Fungsi `sentiment_score()` menjumlahkan skor tiap token, membalik tanda skor apabila terdapat kata negasi dalam jendela, kemudian `score_to_label()` menentukan label akhir: `positive` (skor > 0), `negative` (skor < 0), atau `ambiguous` (skor = 0). Data dengan label `ambiguous` dikeluarkan dari proses pemodelan.

### 4.2 Ekstraksi Fitur dan Proses Training

#### 4.2.1 Ekstraksi Fitur TF-IDF

Fitur diekstraksi menggunakan `TfidfVectorizer` dengan `max_features=150`, menghasilkan matriks representasi numerik dari teks ulasan. Bobot TF-IDF untuk term *t* pada dokumen *d* dihitung dengan rumus:

TF-IDF(t, d) = TF(t, d) × log(N / DF(t))

Di mana TF(t, d) adalah frekuensi kemunculan term *t* dalam dokumen *d*, N adalah jumlah total dokumen, dan DF(t) adalah jumlah dokumen yang mengandung term *t*.

#### 4.2.2 Model Baseline (Logistic Regression)

Model baseline dilatih menggunakan `LogisticRegression(max_iter=1000, class_weight="balanced")` langsung pada matriks TF-IDF tanpa pembobotan tambahan, sebagai pembanding terhadap model yang dioptimasi PSO.

#### 4.2.3 Fungsi Fitness PSO

Fungsi `make_fitness_function()` mendefinisikan nilai fitness untuk setiap kandidat vektor bobot fitur:

fitness(w) = F1_macro_mean(inner-CV) − λ × mean((w − 1)²)

Di mana w adalah vektor bobot fitur yang dibatasi pada rentang [0, 1], F1_macro_mean dihitung dari rata-rata F1-score macro pada validasi silang internal (Repeated Stratified K-Fold), dan λ (REG_LAMBDA) adalah koefisien penalti yang mendorong bobot mendekati nilai netral (1.0), yang secara khusus berada tepat pada batas atas rentang [0, 1] tersebut.

#### 4.2.4 Algoritma Particle Swarm Optimization

Algoritma PSO diimplementasikan pada fungsi `pso_optimize()` dengan tahapan berikut:
1. Inisialisasi swarm partikel dengan posisi (bobot fitur) dan kecepatan acak
2. Evaluasi fitness setiap partikel dan pembaruan *personal best* serta *global best*
3. Pembaruan inertia weight secara linear menurun dari W_MAX ke W_MIN setiap iterasi
4. Pembaruan kecepatan dan posisi partikel berdasarkan komponen kognitif (`c1`) dan sosial (`c2`)
5. Pembatasan kecepatan (`v_max`) dan posisi pada rentang [0, 1], dengan pembalikan arah kecepatan saat posisi keluar batas
6. Reinisialisasi sebagian partikel (`REINIT_FRACTION`) apabila terjadi stagnasi selama `STAGNATION_LIMIT` iterasi berturut-turut, guna menghindari konvergensi prematur pada optimum lokal

### 4.3 Evaluasi Model

#### 4.3.1 Metrik Evaluasi

Evaluasi model dilakukan menggunakan metrik berikut pada data uji setiap fold:

a) **Accuracy**: Accuracy = Σ prediksi benar / N_total_sampel

b) **F1-score (macro)**: rata-rata harmonik precision dan recall yang dihitung per kelas kemudian dirata-ratakan secara sederhana antar kelas, sehingga tidak bias terhadap kelas mayoritas:

F1_macro = (1/C) × Σ F1ᵢ, dengan F1ᵢ = 2 × (Precisionᵢ × Recallᵢ) / (Precisionᵢ + Recallᵢ)

#### 4.3.2 Confusion Matrix

Confusion Matrix berukuran 2x2 (positif/negatif) dihitung menggunakan `confusion_matrix()` dari scikit-learn dan divisualisasikan dengan `ConfusionMatrixDisplay`, baik untuk model baseline maupun model PSO-optimized, guna membandingkan pola kesalahan klasifikasi kedua model.

#### 4.3.3 Skema Repeated Stratified K-Fold

Evaluasi akhir menggunakan `RepeatedStratifiedKFold` dengan 5 fold dan 3 pengulangan (total 15 kali evaluasi) untuk memastikan proporsi kelas tetap terjaga pada setiap fold dan hasil evaluasi lebih stabil dibandingkan satu kali pembagian data (single split). Mode `QUICK_TEST` tersedia untuk menjalankan satu kali split 80/20 guna pengujian cepat kebenaran kode.

### 4.4 Acceptance Gate dan Prediksi Hybrid

Untuk mencegah penggunaan bobot PSO yang tidak benar-benar meningkatkan performa, sistem menerapkan mekanisme **acceptance gate**: fitness bobot hasil PSO dibandingkan dengan fitness bobot netral (seluruh bobot = 1). Bobot PSO hanya diterima apabila selisihnya (`pso_gain`) melebihi ambang `PSO_MIN_IMPROVEMENT` (0.01); jika tidak, sistem menggunakan bobot netral (fallback).

Pada tahap prediksi, fungsi `hybrid_predict()` menggunakan prediksi model PSO-optimized, kecuali untuk sampel dengan margin probabilitas antar kelas yang sangat tipis (kurang dari `PSO_TIE_MARGIN` = 0.03), yang secara otomatis menggunakan prediksi model baseline. Mekanisme ini valid secara metodologis karena hanya memanfaatkan confidence margin dari probabilitas model, tanpa melihat label sebenarnya (tidak terjadi data leakage).

### 4.5 Pengujian Black Box

Pengujian black box dilakukan untuk memverifikasi bahwa sistem berfungsi sesuai spesifikasi tanpa melihat detail implementasi internal.

| Skenario Pengujian | Input | Output yang Diharapkan | Status |
|---|---|---|---|
| Pemuatan dataset dengan encoding berbeda | File CSV dengan encoding non-UTF-8 | Dataset berhasil dibaca melalui fallback encoding | PASS |
| Deteksi kolom teks otomatis | CSV dengan nama kolom bervariasi (mis. "ulasan", "review") | Kolom teks terdeteksi otomatis tanpa error | PASS |
| Data teks kosong/rusak | Baris dengan nilai "nan", "-", atau kosong | Baris tersebut dibuang dari dataset | PASS |
| Pelabelan berbasis leksikon | Teks ulasan dengan kata negasi (mis. "tidak enak") | Skor sentimen berbalik polaritas sesuai negasi | PASS |
| Gagal tulis file (izin ditolak) | File CSV/PNG sedang terbuka di program lain | Sistem menyimpan ke nama file alternatif tanpa crash | PASS |
| Acceptance gate PSO | Peningkatan fitness PSO di bawah ambang | Sistem fallback ke bobot netral | PASS |
| Prediksi hybrid margin tipis | Probabilitas prediksi mendekati 50:50 | Sistem menggunakan prediksi baseline | PASS |
| Evaluasi Repeated Stratified K-Fold | QUICK_TEST = False | Sistem menjalankan 15 kali evaluasi (5 fold x 3 repeat) | PASS |

---

## BAB V PENUTUP

### 5.1 Kesimpulan

Berdasarkan hasil implementasi dan analisis, sistem analisis sentimen ulasan restoran berbasis TF-IDF dan Logistic Regression yang dioptimasi menggunakan Particle Swarm Optimization berhasil dibangun dengan pipeline yang mencakup preprocessing teks Bahasa Indonesia, pelabelan berbasis leksikon, ekstraksi fitur, optimasi bobot fitur, hingga evaluasi menggunakan Repeated Stratified K-Fold. Mekanisme acceptance gate dan prediksi hybrid yang diterapkan memastikan bobot hasil PSO hanya digunakan ketika benar-benar memberikan peningkatan performa yang meyakinkan, sehingga sistem lebih robust dan tidak mudah overfitting terhadap data latih.

### 5.2 Saran

Beberapa saran untuk pengembangan lebih lanjut:
1. Memperkaya kamus leksikon positif dan negatif dengan lebih banyak kosakata dan variasi bahasa gaul/daerah agar pelabelan lebih akurat.
2. Membandingkan performa PSO dengan algoritma metaheuristik lain (mis. Genetic Algorithm atau Ant Colony Optimization) untuk pembobotan fitur.
3. Menambah jumlah data ulasan restoran dari berbagai sumber untuk meningkatkan generalisasi dan kestabilan hasil evaluasi model.
