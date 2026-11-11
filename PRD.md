# Product Requirement Document (PRD) & Technical Specification

## Fitur Klasifikasi Sentimen Ulasan Produk Otomatis (E-Commerce)

| Metadata Dokumen | Deskripsi |
| :--- | :--- |
| **Project Name** | E-Commerce Customer Review Sentiment Analysis Engine |
| **Document Type** | Product Requirement Document (PRD) & Technical Spec |
| **Author / Role** | AI Engineer |
| **Stakeholder** | E-Commerce Product Team, Engineering Lead, Customer Experience Team |
| **Status** | In-Review / Experiment Phase |
| **Version** | 1.1.0 |

---

## 1. Executive Summary & Problem Statement

### 1.1 Latar Belakang & Problem Statement
Tim produk e-commerce menghadapi lonjakan ulasan pelanggan harian pada halaman katalog produk. Saat ini, tim tidak memiliki mekanisme otomatis yang andal untuk memisahkan feedback pelanggan yang puas dari keluhan kritis pelanggan. Analisis ulasan secara manual membutuhkan biaya operasional tinggi, lambat, dan tidak dapat diskalakan seiring pertumbuhan transaksi harian.

Untuk mengatasi isu ini, tim produk membutuhkan fitur otomasi berbasis AI guna mengklasifikasikan sentimen setiap ulasan pelanggan secara *real-time* atau *near-real-time*. Sebelum sistem diterapkan ke level produksi, tim AI Engineer ditugaskan melakukan studi kelayakan dan eksperimen komparasi antara dua kandidat pendekatan:
1. **Pendekatan 1 — Model Klasik Machine Learning (In-house Trained)**: Model berbasis ekstraksi fitur statistik (TF-IDF) dan algoritma supervised learning klasik (Logistic Regression) menggunakan Scikit-learn.
2. **Pendekatan 2 — Large Language Model (Cloud API-based)**: Model penalaran bahasa terpusat via Google Gemini AI API menggunakan pendekatan prompt engineering tanpa proses training bobot model (*zero-shot/few-shot in-context learning*).

### 1.2 Objective (Tujuan Bisnis & Teknis)
* **Objective Produk:** Menyediakan modul backend yang mampu mengklasifikasikan ulasan pelanggan ke dalam kategori sentimen secara otomatis, akurat, dan konsisten.
* **Objective Rekayasa AI:** Melakukan benchmarking empiris antara Classical ML dan LLM API pada dataset pengujian yang identik (*apple-to-apple comparison*), mengukur metrik performa klasifikasi, latensi, efisiensi biaya (*Total Cost of Ownership*), serta menyusun rekomendasi teknis berbasis data.

---

## 2. Definisi Target, Label, & Asumsi

### 2.1 Skema Target Prediksi
Sistem klasifikasi difokuskan pada problem klasifikasi biner (*binary sentiment classification*):
* **`positif`**: Ulasan yang mengekspresikan kepuasan, apresiasi kualitas barang, kecepatan pengiriman, atau rekomendasi produk.
* **`negatif`**: Ulasan yang memuat keluhan cacat produk, keterlambatan pengiriman, kekecewaan layanan, atau ketidaksesuaian deskripsi barang.

### 2.2 Batasan & Asumsi Sistem (Constraints & Assumptions)
1. **Bahasa Korpus:** Data ulasan berbahasa Indonesia (termasuk bahasa informal, slang pasar, singkatan, dan tanda baca bebas).
2. **Kondisi Dataset:** Dataset eksperimen berukuran 200 baris data ulasan berlabel pada file `customer_reviews_sentiment.csv`. File asli di folder `data/` tidak boleh diubah secara langsung.
3. **Integritas Test Set:** Pembagian data wajib menggunakan rasio 80% *training set* (160 data) dan 20% *test set* (40 data) dengan teknik *stratified sampling* (`random_state=42`, `stratify=y`).
4. **Validitas Komparasi (Fair Evaluation):** Evaluasi performa model Scikit-Learn dan Gemini API **wajib dieksekusi pada 40 sampel data test yang sama persis** untuk mencegah bias data.
5. **Kerahasiaan Kredensial:** API Key Gemini wajib dimuat melalui Environment Variable (`.env` / `GEMINI_API_KEY`) dan dilarang di-*hardcode* di dalam kode. File `.env` wajib dimasukkan ke dalam `.gitignore`.

---

## 3. User Persona & User Journey

### 3.1 Target Pengguna
1. **Product Manager E-Commerce:** Memantau metrik kepuasan produk (CSAT) secara agregat melalui dashboard analitik.
2. **Customer Support Specialist:** Mendapatkan notifikasi prioritas instan saat ada ulasan berlabel `negatif` dari pelanggan agar penanganan keluhan dapat dilakukan sebelum rating toko turun.
3. **Calon Pembeli (End-User):** Mendapatkan filter ulasan produk berdasarkan sentimen untuk membantu keputusan pembelian.

### 3.2 Alur Sistem Produksi Konseptual (User Journey)
```
[Pelanggan Submit Ulasan] 
          │
          ▼
[API Gateway / Backend Service]
          │
          ▼
[AI Sentiment Inference Engine] ──► (Prediksi: 'positif' / 'negatif')
          │
          ├──► Update Status di Database & Halaman Produk
          └──► (Jika Negatif) Trigger Alert ke CS Helpdesk
```

---

## 4. Spesifikasi Eksperimen & Desain Teknis

### 4.1 Pendekatan 1 — Classical Machine Learning (Scikit-Learn)
* **Pipeline:**
  1. *Feature Extraction:* `TfidfVectorizer()` (konversi teks kalimat menjadi representasi bobot TF-IDF). Menggunakan `.fit_transform(X_train)` dan `.transform(X_test)` guna mencegah *data leakage*.
  2. *Classifier:* `LogisticRegression(random_state=42)` (pembentukan hyperplane pemisah probabilitas biner).
* **Karakteristik Operasional:**
  * Komputasi inferensi terjadi 100% lokal pada CPU server backend.
  * Latensi inferensi ultra-rendah (<10 ms).
  * Biaya inferensi Rp0 (tidak ada konsumsi kuota API pihak ketiga).

### 4.2 Pendekatan 2 — Large Language Model via API (Google Gemini)
* **Engine:** Google GenAI SDK (`gemini-2.5-flash` / `gemini-2.0-flash`).
* **Strategi Prompting:** *Zero-shot Prompting* dengan strict schema guardrails.
  * **System Instruction / Template:**
    ```
    Klasifikasikan sentimen ulasan pelanggan e-commerce berikut sebagai "positif" atau "negatif".
    Jawab HANYA dengan satu kata dalam huruf kecil: positif atau negatif.
    Dilarang menambahkan tanda baca, pengantar, atau penjelasan tambahan.

    Ulasan: "{review_text}"
    Sentimen:
    ```
* **Hyperparameter Control:**
  * `temperature: 0.0` (atau `0.2`): Meminimalkan keacakan pemilihan token demi menghasilkan respons deterministik (*greedy decoding*).
  * `max_output_tokens: 5`: Membatasi panjang keluaran guna menekan biaya dan mencegah model merespons berlebihan.
* **Post-Processing Guardrail:**
  * Normalisasi string wajib: `.strip().lower().replace('"', '').replace('.', '')` untuk menjamin kesesuaian nilai kategori target (`positif` / `negatif`).

---

## 5. Metrik Evaluasi & Kriteria Keberhasilan (Success Metrics)

### 5.1 Metrik Kuantitatif (Machine Learning Metrics)
Kedua pendekatan dievaluasi terhadap ground truth `y_test` dengan metrik berikut:
1. **Accuracy:** Persentase kebenaran total prediksi di seluruh data uji.
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Positif & Negatif):** Mengukur rasio ketepatan prediksi pada kelas tertentu guna meminimalkan *false alarm*.
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Positif & Negatif):** Mengukur kemampuan sistem mendeteksi keseluruhan entri sentimen yang sebenarnya ada di data uji.
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score (Macro / Weighted):** Rata-rata harmonik antara precision dan recall.
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Confusion Matrix:** Matriks kontingensi $2 \times 2$ pemetaan True Positive, False Positive, True Negative, dan False Negative.

### 5.2 Metrik Kualitatif & Rekayasa Sistem (Engineering Constraints)
| Metrik Teknis | Target / Toleransi Minimum |
| :--- | :--- |
| **Minimum Test Accuracy** | $\ge 85.0\%$ pada test set |
| **Inference Latency (SLA)** | Klasik: $< 20\text{ ms}$ \| LLM API: $< 1.500\text{ ms}$ per request |
| **Output Consistency** | 100% output terpetakan ke enum `['positif', 'negatif']` tanpa parsing error |
| **Availability / Downtime** | Klasik: 99.99% (internal host) \| API: Bergantung SLA vendor cloud |

---

## 6. Kerangka Analisis Trade-off (Decision Matrix)

Eksperimen ini akan menilai kelayakan produksi berdasarkan 5 pilar arsitektur:
```
                      ┌─────────────────────────────────┐
                      │    DIMENSI ANALISIS TRADE-OFF   │
                      └────────────────┬────────────────┘
                                       │
      ┌───────────────┬────────────────┼────────────────┬───────────────┐
      ▼               ▼                ▼                ▼               ▼
[ Performance ]  [ Latency ]     [ Cost / TCO ]   [ Complexity ]  [ Flexibility ]
Akurasi semantik Kecepatan waktu  Biaya per-kueri  Effort training  Ketahanan slang
& deteksi slang  respons inferensi vs biaya infra  & pipeline MLOps & konteks baru
```

---

## 7. Project Milestones, Deliverables, & Acceptance Criteria

Pelaksanaan proyek dipecah ke dalam 5 milestone kerja berurutan yang memetakan tahapan tugas, kriteria rubrik penilaian (total 100%), dan acceptance criteria untuk eksekusi agent:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ROADMAP MILESTONE EKSEKUSI                            │
└─────────────────────────────────────────────────────────────────────────────┘
  [ M1: Prepare ] ──► [ M2: Classic ML ] ──► [ M3: LLM API ] ──► [ M4: Evaluate ] ──► [ M5: Recommend ]
  (Bobot: 10%)        (Bobot: 20%)           (Bobot: 20%)        (Bobot: 25%)         (Bobot: 25%)
```

### Milestone 1: Environment Setup & Problem Statement (Prepare — Bobot 10%)
* **Tujuan:** Menyiapkan struktur repositori lokal, konfigurasi dependensi dan API key, serta mendokumentasikan pemahaman problem statement.
* **Deliverables:**
  * Struktur repositori terstandardisasi (`data/`, `notebook/`, `documentation/`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`).
  * File dataset asli `customer_reviews_sentiment.csv` tersimpan aman di `data/`.
  * Section 1 di `experiment_notebook.ipynb` dan bagian awal `README.md` terisi narasi problem statement.
* **Acceptance Criteria:**
  * [ ] Semua dependensi (`pandas`, `scikit-learn`, `google-genai`, `python-dotenv`, `matplotlib`, `seaborn`) terpasang tanpa bentrok versi.
  * [ ] File `.env` memuat `GEMINI_API_KEY` dan nama file `.env` tercantum di `.gitignore` (proteksi dari kebocoran ke GitHub).
  * [ ] Problem statement mendefinisikan secara eksplisit:
    * Objective bisnis dan teknis.
    * Target klasifikasi biner (`positif` / `negatif`).
    * Batasan bahasa Indonesia, ukuran data, dan aturan test set identik.

### Milestone 2: Data Preprocessing & Classical ML Training (Experiment Part 1 — Bobot 20%)
* **Tujuan:** Membagi dataset secara stratified dan membangun baseline model klasifikasi teks Scikit-learn.
* **Deliverables:**
  * Kode load dataset dan train-test split di Section 2 & 3 notebook.
  * Pipeline `TfidfVectorizer` dan `LogisticRegression` di Section 4 notebook.
  * Variabel output prediksi `y_pred_classic`.
* **Acceptance Criteria:**
  * [ ] Data terbagi dengan rasio 80:20 (160 train, 40 test) menggunakan `random_state=42` dan `stratify=y`.
  * [ ] Vektorisasi mematuhi aturan baku ML: `.fit_transform(X_train)` dan `.transform(X_test)` (zero data leakage).
  * [ ] Model `LogisticRegression` berhasil fit dan menghasilkan 40 label prediksi biner pada `X_test`.

### Milestone 3: LLM Inference via Gemini API (Experiment Part 2 — Bobot 20%)
* **Tujuan:** Merancang prompt classification deterministik dan mengeksekusi batch inference Gemini pada test set yang identik.
* **Deliverables:**
  * Inisialisasi client SDK `google.genai` berbasis environment variable di Section 5.1 notebook.
  * Fungsi prompt engineering `classify_sentiment_llm()` di Section 5.2 notebook.
  * Loop inferensi dengan sanitasi string dan variabel `y_pred_llm` di Section 5.3 notebook.
* **Acceptance Criteria:**
  * [ ] API client terhubung aman tanpa hardcoded key.
  * [ ] Prompt memiliki instruksi pembatas yang ketat ("Jawab HANYA satu kata: positif atau negatif").
  * [ ] Hyperparameter didefinisikan secara sadar (`temperature=0.0` atau `0.2`) dengan dokumentasi rasionalisasinya di notebook.
  * [ ] Inferensi dijalankan tepat pada 40 baris `X_test` yang sama dengan Milestone 2.
  * [ ] Output diproses dengan pembersihan string (`.strip().lower()`) sehingga 100% prediksi konsisten bernilai `'positif'` atau `'negatif'`.

### Milestone 4: Model Evaluation & Comparative Analysis (Evaluate — Bobot 25%)
* **Tujuan:** Mengukur performa kuantitatif kedua pendekatan dan menyusun visualisasi serta matriks komparasi.
* **Deliverables:**
  * Output `accuracy_score`, `classification_report`, dan `confusion_matrix` untuk kedua model di Section 6 notebook.
  * DataFrame / Markdown table ringkasan perbandingan metrik.
  * Visualisasi Confusion Matrix disimpan ke `documentation/model_comparison_summary.png`.
* **Acceptance Criteria:**
  * [ ] Seluruh metrik (Accuracy, Precision, Recall, F1-Score) dihitung secara valid menggunakan `sklearn.metrics`.
  * [ ] Confusion matrix kedua model berhasil dibuat dan disertai analisis True Positive, False Positive, dan False Negative.
  * [ ] Setiap angka metrik memiliki interpretasi kontekstual (bukan hanya menampilkan tabel angka mentah).

### Milestone 5: Trade-off Analysis, Technical Recommendation & Submission (Recommend — Bobot 25%)
* **Tujuan:** Menyusun analisis teknis-bisnis, merumuskan rekomendasi arsitektur produksi bertahap, dan mempublikasikan repositori submission.
* **Deliverables:**
  * Analisis Trade-off & Limitations di Section 7 notebook dan `README.md`.
  * Rekomendasi Technical Approach di Section 8 notebook dan `README.md`.
  * File `requirements.txt` dan `README.md` final yang lengkap dan rapi.
  * Repositori GitHub publik dengan format nama sesuai penugasan.
* **Acceptance Criteria:**
  * [ ] Analisis trade-off membedah secara mendalam 4 pilar:
    1. *Effort Implementasi:* Kebutuhan data berlabel & training pipeline vs prompting instan tanpa training.
    2. *Latency & Throughput:* Komputasi lokal kilat (<10 ms) vs overhead network API (~300–800 ms).
    3. *Cost Structure:* Fixed CPU infra cost vs variable per-token API cost.
    4. *Language Nuance:* Keterbatasan n-gram TF-IDF vs penalaran semantik LLM pada slang/sarkasme.
  * [ ] Rekomendasi technical approach memuat skenario adopsi berbasis fase (misal: Gemini API untuk fase MVP/validasi cepat; migrasi ke Classical ML / fine-tuned saat volume transaksi harian masif).
  * [ ] Kernel notebook berhasil dijalankan ulang dari awal (*Restart and Run All Cells*) tanpa ada error runtime.
  * [ ] Repository GitHub dibuat dengan nama `assignment-model-experiment-<nama-peserta>` dan disetel publik.