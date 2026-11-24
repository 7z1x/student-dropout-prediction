# 🎓 Student Dropout Early-Warning System — Jaya Jaya Institut

Mendeteksi mahasiswa berisiko **dropout** sejak dini menggunakan machine learning, lalu menerjemahkannya menjadi rekomendasi intervensi yang bisa dieksekusi oleh staf akademik.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen?logo=xgboost)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://drop-out-predict.streamlit.app/)
[![Metabase](https://img.shields.io/badge/Dashboard-Metabase-509EE3?logo=metabase&logoColor=white)](https://www.metabase.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88.47%25-success)]()

**🔗 Live Demo:** [drop-out-predict.streamlit.app](https://drop-out-predict.streamlit.app/) · **📊 Dashboard:** Metabase · **🎬 Video:** [Google Drive](https://drive.google.com/file/d/1tGv-VZaOavT4nUHSdPltAfbsRvWfl0Oy/view?usp=drive_link)

---

## 📌 Ringkasan Eksekutif

| Pertanyaan | Jawaban |
|---|---|
| Berapa besar masalahnya? | **32,1%** mahasiswa (1.421 dari 4.424) berstatus *dropout* |
| Bisa diprediksi? | Ya — akurasi **88,47%** pada data uji, **recall 79%** untuk kelas dropout |
| Apa pemicu terkuat? | Jumlah mata kuliah lulus semester 2, status pembayaran UKT, dan status *debtor* |
| Apa tindak lanjutnya? | Sistem skoring risiko + dashboard monitoring + SOP intervensi per profil risiko |

> **Insight utama:** dropout bukan kejadian mendadak, melainkan proses yang terlihat dari data. Mahasiswa yang dropout rata-rata hanya lulus **1,94 mata kuliah** di semester 2 (vs **6,18** pada mahasiswa lulus), dan **86,6%** mahasiswa yang menunggak UKT akhirnya dropout.

---

## 🎯 Business Understanding

### Latar Belakang

Jaya Jaya Institut berdiri sejak tahun 2000 dan memiliki reputasi baik, namun masih menghadapi tingkat *dropout* yang tinggi. Setiap mahasiswa yang dropout adalah kerugian ganda: hilangnya potensi lulusan **dan** hilangnya pendapatan UKT yang sudah diinvestasikan institusi.

Selama ini institusi bersifat **reaktif** — mahasiswa baru ditangani setelah mereka berhenti. Yang dibutuhkan adalah sistem **proaktif**: mengenali sinyal risiko lebih awal, sehingga bimbingan, keringanan biaya, atau konseling bisa diberikan sebelum keputusan dropout diambil.

### Permasalahan Bisnis

1. **Deteksi terlambat** — belum ada mekanisme sistematis untuk menandai mahasiswa berisiko sebelum semester berjalan terlalu jauh.
2. **Intervensi tidak terarah** — bantuan diberikan merata, padahal risiko terkonsentrasi pada segmen tertentu (mis. mahasiswa dengan tunggakan UKT).
3. **Tidak ada ukuran dampak** — belum ada dashboard yang memantau tren dropout per program studi secara berkala.

### Tujuan Proyek

| # | Tujuan | Output |
|---|---|---|
| 1 | Mengidentifikasi faktor paling berpengaruh terhadap dropout | Analisis EDA + feature importance |
| 2 | Membangun model prediksi risiko dropout | Model XGBoost (`.joblib`) |
| 3 | Menyediakan alat skoring untuk staf akademik | Aplikasi Streamlit |
| 4 | Memantau tren & mendukung keputusan | Dashboard Metabase |

### Metrik Keberhasilan

- **Metrik bisnis:** penurunan persentase dropout per angkatan; jumlah mahasiswa berisiko tinggi yang menerima intervensi tepat waktu.
- **Metrik model:** **Recall** pada kelas *dropout* diprioritaskan di atas akurasi, karena biaya *false negative* (mahasiswa berisiko tidak terdeteksi) jauh lebih besar daripada *false positive* (mahasiswa aman ikut dibina).

### Cakupan

**Termasuk:** analisis data historis, pemodelan prediktif, prototipe aplikasi, dashboard bisnis, rekomendasi aksi.
**Tidak termasuk:** integrasi langsung ke Sistem Informasi Akademik, uji coba A/B pada populasi mahasiswa nyata, analisis data kualitatif.

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | [Dicoding Dataset — Students' Performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance) |
| **Jumlah baris** | 4.424 mahasiswa |
| **Jumlah fitur** | 36 fitur + 1 target |
| **Missing value** | 0 |
| **Duplikat** | 0 |
| **Periode** | Data historis pendaftaran & performa akademik |

**Target (`Status`) — distribusi tidak seimbang:**

| Status | Jumlah | Proporsi |
|---|---:|---:|
| Graduate | 2.209 | 49,9% |
| **Dropout** | **1.421** | **32,1%** |
| Enrolled | 794 | 18,0% |

**Kelompok fitur:**

- **Demografi & pendaftaran:** usia saat masuk, gender, status pernikahan, kewarganegaraan, mahasiswa internasional, disabilitas
- **Latar belakang keluarga:** kualifikasi & pekerjaan ibu/ayah
- **Akademik:** jalur masuk, program studi, nilai kualifikasi sebelumnya, nilai admisi
- **Status keuangan:** tunggakan (`Debtor`), kelancaran UKT (`Tuition_fees_up_to_date`), penerima beasiswa
- **Performa kuliah:** unit kurikulum semester 1 & 2 (diambil, dievaluasi, lulus, nilai rata-rata)
- **Makroekonomi:** tingkat pengangguran, inflasi, GDP

---

## 🔬 Metodologi

```
Data Loading → EDA → Data Preparation → SMOTE → XGBoost + Tuning → Evaluasi + Deployment
```

1. **Data Loading** — memuat `dataset/data.csv` (delimiter `;`): 4.424 baris, 36 fitur.
2. **EDA** — distribusi target, korelasi fitur vs target, boxplot outlier, cek missing value.
3. **Data Preparation** — split **80:20 stratified** (3.539 train / 885 test), lalu `StandardScaler`.
4. **SMOTE** — menyeimbangkan kelas pada data latih: 2.402 → 4.804 baris (50:50).
5. **Modeling** — `XGBClassifier` + `RandomizedSearchCV` (50 kombinasi, 5-fold CV, scoring `f1`).
6. **Evaluasi & Deployment** — confusion matrix, precision/recall/F1, lalu export model ke Streamlit + Metabase.

> **Catatan penting:** `StandardScaler` dan SMOTE hanya diterapkan pada **data latih**. Data uji dibiarkan pada distribusi aslinya supaya evaluasi tidak bocor (*data leakage*) dan tetap jujur.

**Hyperparameter terbaik:**

| Parameter | Nilai |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 9 |
| `learning_rate` | 0,0448 |
| `subsample` | 0,849 |
| `colsample_bytree` | 0,630 |
| `gamma` | 0,179 |

**Mengapa XGBoost?** Mampu menangkap interaksi non-linear antar fitur, tahan terhadap outlier dan fitur berskala berbeda, menangani data tabular dengan baik, serta menyediakan *feature importance* yang bisa diterjemahkan menjadi insight bisnis.

---

## 📊 Temuan Utama (EDA)

### 1. Performa akademik semester awal adalah sinyal terkuat

| Metrik (rata-rata) | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Nilai semester 1 | 7,26 | 11,13 | 12,64 |
| Nilai semester 2 | **5,90** | 11,12 | **12,70** |
| Mata kuliah lulus semester 1 | 2,55 | 4,32 | 6,23 |
| Mata kuliah lulus semester 2 | **1,94** | 4,06 | **6,18** |

Selisih nilai semester 2 antara mahasiswa dropout dan lulus mencapai **~6,8 poin** — jurang yang sangat lebar dan terlihat jelas sejak awal.

### 2. Kesulitan finansial berkorelasi kuat dengan dropout

| Segmen | Tingkat Dropout | Selisih |
|---|---:|---|
| UKT **tidak** lancar | **86,6%** | vs 24,7% (lancar) — **3,5×** |
| Berstatus **debtor** | **62,0%** | vs 28,3% — **2,2×** |
| **Bukan** penerima beasiswa | **38,7%** | vs 12,2% (penerima) — **3,2×** |

Tunggakan UKT adalah **satu-satunya faktor non-akademik** dengan daya pisah sekuat ini.

### 3. Gender menunjukkan perbedaan bermakna

Tingkat dropout mahasiswa laki-laki **45,1%** vs perempuan **25,1%** — perlu ditelusuri lebih lanjut apakah ini efek komposisi program studi.

### 4. Faktor yang ternyata **tidak** signifikan

Status mahasiswa internasional hampir tidak berpengaruh: **29,1%** (internasional, n=110) vs **32,2%** (lokal, n=4.314). Ukuran sampel internasional terlalu kecil untuk dijadikan dasar kebijakan — klaim ini sengaja tidak dimasukkan ke rekomendasi.

### 5. Korelasi terhadap status dropout

| Fitur | Korelasi \|r\| |
|---|---:|
| `Curricular_units_2nd_sem_grade` | 0,572 |
| `Curricular_units_2nd_sem_approved` | 0,570 |
| `Curricular_units_1st_sem_grade` | 0,481 |
| `Curricular_units_1st_sem_approved` | 0,479 |
| `Tuition_fees_up_to_date` | 0,429 |

---

## 🤖 Hasil Model

### Performa pada Data Uji (n = 885)

| Metrik | Nilai |
|---|---:|
| **Accuracy** | **88,47%** |
| Macro avg Precision | 0,87 |
| Macro avg Recall | 0,86 |
| Macro avg F1-score | 0,87 |

**Per kelas:**

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Tidak Dropout (0) | 0,90 | 0,93 | 0,92 | 601 |
| **Dropout (1)** | **0,84** | **0,79** | **0,81** | 284 |

**Confusion matrix:**

|  | Prediksi: Tidak Dropout | Prediksi: Dropout |
|---|---:|---:|
| **Aktual: Tidak Dropout** | 560 (TN) | 41 (FP) |
| **Aktual: Dropout** | 61 (FN) | 223 (TP) |

**Cara membaca untuk manajemen:** dari 284 mahasiswa yang benar-benar dropout, model berhasil menandai **223 orang (79%)**. Sisa **61 orang (21%) terlewat** — inilah kelompok yang harus ditangkap melalui jalur non-model (mis. laporan dosen wali). Dari 601 mahasiswa yang sebenarnya aman, **41 orang (7%) salah ditandai** — biaya kesalahannya rendah karena hanya berupa bimbingan tambahan yang tidak merugikan.

### 10 Fitur Paling Berpengaruh

| # | Fitur | Importance |
|---:|---|---:|
| 1 | `Curricular_units_2nd_sem_approved` | 0,220 |
| 2 | `Tuition_fees_up_to_date` | 0,166 |
| 3 | `Curricular_units_1st_sem_approved` | 0,043 |
| 4 | `Debtor` | 0,037 |
| 5 | `Curricular_units_1st_sem_enrolled` | 0,032 |
| 6 | `Gender` | 0,028 |
| 7 | `Scholarship_holder` | 0,028 |
| 8 | `Curricular_units_2nd_sem_enrolled` | 0,027 |
| 9 | `Curricular_units_2nd_sem_credited` | 0,024 |
| 10 | `Curricular_units_2nd_sem_grade` | 0,024 |

Dua fitur teratas menyumbang **~39%** total importance: **jumlah mata kuliah lulus di semester 2** dan **status pembayaran UKT**. Keduanya bisa dipantau langsung oleh sistem akademik — tidak perlu kuesioner atau data tambahan.

---

## ⚠️ Keterbatasan & Asumsi

Bagian ini penting untuk menilai kesiapan model sebelum dipakai di produksi:

1. **Risiko keterlambatan deteksi.** Fitur terkuat berasal dari **semester 2**, artinya model paling akurat setelah mahasiswa menempuh hampir satu tahun. Untuk intervensi yang benar-benar dini, perlu model terpisah yang hanya memakai data semester 1 / saat pendaftaran (trade-off: akurasi lebih rendah, tetapi *lead time* lebih panjang).
2. **Evaluasi single split.** Metrik berasal dari satu *train-test split* (80:20). Belum ada validasi silang berulang atau uji pada angkatan berbeda, sehingga angka 88,47% bisa sedikit optimistis.
3. **Belum ada validasi temporal.** Idealnya model dilatih pada angkatan lama dan diuji pada angkatan terbaru untuk meniru kondisi pemakaian nyata.
4. **SMOTE pada data yang sudah di-scale.** Praktik yang lebih aman adalah menerapkan SMOTE sebelum scaling atau memakai `Pipeline` agar seluruh transformasi ter-*fit* di dalam CV.
5. **Ambang keputusan masih default (0,5).** Untuk kasus dropout, ambang lebih rendah (mis. 0,35) akan menaikkan recall dan menangkap lebih banyak mahasiswa berisiko — perlu dikalibrasi bersama tim akademik sesuai kapasitas intervensi.
6. **Korelasi ≠ kausalitas.** Feature importance menunjukkan asosiasi, bukan sebab-akibat. Rekomendasi di bawah adalah hipotesis yang perlu diuji lewat pilot program.

---

## 💡 Rekomendasi Action Items

Disusun berdasarkan prioritas **dampak × kemudahan eksekusi**.

### Prioritas 1 — Sistem Peringatan Dini Akademik (dampak tinggi, cepat)

- **Aktifkan skoring risiko tiap akhir semester**, fokus pada indikator `Curricular_units_2nd_sem_approved` dan nilai semester.
- **Aturan triage sederhana:** mahasiswa dengan **< 3 mata kuliah lulus** di semester mana pun otomatis masuk daftar pantau dosen pembimbing akademik (DPA) — selaras dengan temuan bahwa rata-rata mahasiswa dropout hanya lulus 1,94 MK.
- **Bekali DPA dengan skor risiko** mahasiswa bimbingannya; wajibkan minimal satu sesi konseling terstruktur untuk kelompok risiko tinggi.

### Prioritas 2 — Deteksi & Intervensi Finansial (dampak tinggi)

- **Picu notifikasi otomatis** ke unit kemahasiswaan saat mahasiswa masuk status `Debtor` atau UKT tidak lancar — segmen ini memiliki dropout rate **86,6%**, tertinggi di seluruh dataset.
- **Sediakan skema cicilan / penundaan bersyarat / beasiswa darurat** dengan evaluasi per kasus, sebelum tunggakan berlanjut ke semester berikutnya.
- **Perluas cakupan beasiswa** — penerima beasiswa hanya mengalami dropout 12,2%, menjadikannya intervensi dengan rasio manfaat-biaya terbaik.

### Prioritas 3 — Program Dukungan Berbasis Profil (dampak menengah)

- **Mentoring untuk mahasiswa laki-laki** dan program studi dengan dropout rate tertinggi (identifikasi via dashboard), mengingat gap gender 45,1% vs 25,1%.
- **Dukungan khusus mahasiswa dewasa/mandiri** (usia masuk rata-rata 23,3 tahun, maksimum 70) yang perlu menyeimbangkan kuliah dengan pekerjaan dan keluarga.
- **Layanan konseling psikososial** yang peka gender dan inklusif.

### Prioritas 4 — Budaya Keputusan Berbasis Data (jangka panjang)

- **Jadikan dashboard Metabase sebagai rujukan rutin** di level rektorat, dekanat, dan ketua program studi untuk memantau tren dropout per semester.
- **Latih staf** (DPA, admin, tim konseling) membaca dashboard dan menginterpretasikan skor risiko.
- **Integrasikan output model ke Sistem Informasi Akademik / LMS** agar daftar mahasiswa berisiko muncul otomatis, bukan lewat file manual.

### Prioritas 5 — Tata Kelola & Evaluasi

- **Bentuk gugus tugas lintas fungsi** penurunan angka dropout (pimpinan, fakultas, DPA, kemahasiswaan, perwakilan mahasiswa).
- **Evaluasi dampak per semester:** bandingkan dropout rate kelompok yang diintervensi vs kontrol, ukur *lift* dari program.
- **Kumpulkan umpan balik kualitatif** dari mahasiswa berisiko dan staf untuk menyempurnakan SOP intervensi.

### KPI yang Disarankan

| KPI | Baseline | Target (1 tahun) |
|---|---|---|
| Dropout rate keseluruhan | 32,1% | < 28% |
| Dropout pada mahasiswa dengan tunggakan UKT | 86,6% | < 70% |
| Mahasiswa berisiko tinggi yang menerima intervensi | — | ≥ 80% |
| Waktu dari deteksi risiko → intervensi pertama | — | ≤ 14 hari |

---

## 📈 Business Dashboard (Metabase)

Dashboard terhubung ke database dan menyajikan:

- **Ringkasan umum** — total mahasiswa, proporsi dropout, distribusi status
- **Faktor risiko** — perbandingan rata-rata nilai, gender, status UKT
- **Analisis per program studi & status pernikahan** — 5 prodi dengan dropout tertinggi
- **Daftar mahasiswa berisiko tinggi** — untuk tindak lanjut langsung

**Cuplikan layar:**

| Ringkasan Umum | Faktor Risiko |
|---|---|
| ![Dashboard 1](Insight/zulfahmi17z_dashboard.jpg) | ![Dashboard 2](Insight/zulfahmi17z_dashboard%281%29.jpg) |

| Per Prodi & Status Pernikahan | Daftar Mahasiswa Berisiko |
|---|---|
| ![Dashboard 3](Insight/zulfahmi17z_dashboard%282%29.jpg) | ![Dashboard 4](Insight/zulfahmi17z_dashboard%284%29.jpg) |

**Akun demo Metabase** (instance lokal, hanya untuk keperluan demonstrasi):

- Email: `root@mail.com`
- Password: `root123`

> ⚠️ Kredensial di atas adalah untuk instance demo lokal. Jangan gunakan ulang di lingkungan produksi.

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python **3.10+** — cek dengan `python --version`

### 2. Clone & Setup

```bash
git clone https://github.com/7z1x/student-dropout-prediction.git
cd student-dropout-prediction

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Prediksi (Streamlit)

```bash
streamlit run app.py
```

Buka `http://localhost:8501`, isi form data mahasiswa, lalu klik **Predict Status** untuk melihat status prediksi beserta probabilitas tiap kelas.

Aplikasi membutuhkan tiga artefak di folder `model/`:

- `dropout_model.joblib` — model XGBoost terlatih
- `scaler.joblib` — `StandardScaler` hasil training
- `feature_order.joblib` — urutan kolom fitur (wajib sama dengan saat training)

### 4. Uji Prediksi via CLI

```bash
python test_predict.py
```

### 5. Reproduksi Analisis

Buka `notebook.ipynb` (disarankan lewat Google Colab atau Jupyter) untuk menjalankan ulang seluruh pipeline: EDA → preparation → modeling → evaluasi → export model.

---

## 🗂️ Struktur Proyek

```
student-dropout-prediction/
│
├── Insight/                          # Screenshot dashboard & video presentasi
│   ├── zulfahmi17z_dashboard.jpg
│   ├── zulfahmi17z_dashboard(1..4).jpg
│   └── zulfahmi17z_video.mp4
│
├── dataset/
│   └── data.csv                      # 4.424 baris, delimiter ';'
│
├── model/                            # Artefak model terlatih
│   ├── dropout_model.joblib
│   ├── scaler.joblib
│   └── feature_order.joblib
│
├── app.py                            # Aplikasi prediksi Streamlit
├── test_predict.py                   # Skrip prediksi via CLI
├── notebook.ipynb                    # EDA, modeling, evaluasi
├── metabase.db.mv.db                 # Database Metabase (H2)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Analisis data | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Visualisasi | Matplotlib, Seaborn |
| Deployment model | Streamlit |
| Business intelligence | Metabase |
| Model persistence | joblib |

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523

Proyek ini disusun sebagai bagian dari pembelajaran Data Science / Data Analyst.

---

## 📄 Lisensi & Atribusi

Dataset berasal dari [Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset) (UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success*). Kode dalam repositori ini bebas digunakan untuk keperluan pembelajaran.
