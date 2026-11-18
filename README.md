# Klasifikasi Citra Lesi Kulit dengan CNN dan RegNet

## Ringkasan
Studi kasus mata kuliah **Infrastruktur dan Platform untuk Sains Data** menggunakan [Mpox Skin Lesion Dataset v2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data). Proyek ini membandingkan CNN baseline dari materi, CNN dengan arsitektur yang dimodifikasi, dan RegNet untuk mengklasifikasikan enam kelas citra kulit (`Chickenpox`, `Cowpox`,`HFMD`,`Healthy`,`Measles`, dan `Monkeypox`). Dataset ini punya 755 gambar asli yang sudaah dibagi menjadi 5 fold (Train, Valid, Test). Setiap fold juga memiliki sekitar 7500 gambar augmentasi untuk training. Semua gambar kemudian di-resize menjadi 100 x 100.

|Model|Arsitektur|Parameter|
|---|---|---|
|`cnn_baseline`|2 blok Conv-ReLU-Maxpool (20, 50, kernel 5) -> Flatten -> Linear.|Sekitar 51 rb|
|`cnn_modified`|3 blook Conv-BatchNorm-ELU-MaxPool (32-64-128, kernel 3) -> Global Avg Pool -> Dropout 0.4 -> Linear 128 -> ELU -> Dropout 0.3 -> Linear.|Sekitar 111rb|
|`regnet_y_400mf`|RegNetY-400MF [1] pretrained ImageNet, layer terakhir diganti 6 kelas.|Sekitar 3,9jt|

**Status per 17 November 2026:** hasil lima fold untuk dua CNN sudah tersedia dan diverifikasi ulang dari checkpoint lokal. Swin sedang dijalankan oleh penulis; hasilnya belum dimasukkan. Repositori ini belum menyatakan eksperimen tiga model selesai.

**Catatan hasil awal:** kode yang menyertai eksperimen CNN modifikasi menggunakan Augmented Train saja. Versi training di repositori ini telah diperbaiki menjadi Original Train + Augmented Train sesuai tugas. Angka pada `results/initial/` tetap merupakan hasil lama, bukan hasil pelatihan ulang kode yang diperbaiki. Lihat [riwayat dan batas interpretasi](docs/experiment_history.md).

| Model | Accuracy | F1 weighted | ROC-AUC macro OvR |
| --- | --- | --- | --- |
| CNN baseline | 43,47 ± 9,72% | 36,37 ± 8,08% | 0,7408 ± 0,0384 |
| CNN modifikasi | 51,81 ± 5,48% | 44,30 ± 7,74% | 0,7607 ± 0,0350 |
| **RegNetY-400MF** | **75,85 ± 5,80%** | **75,11 ± 5,90%** | **0,9335 ± 0,0110** |

## Tujuan dan alasan pemilihan model

Tujuan eksperimen adalah menilai kemampuan model membedakan Chickenpox, Cowpox, HFMD, Healthy, Measles, dan Monkeypox pada dataset yang ditentukan. CNN baseline menjadi pembanding terhadap implementasi materi. CNN modifikasi menguji kombinasi tiga blok konvolusi, BatchNorm, ELU, pooling global, dan dropout. Swin-Tiny memberi pembanding berbasis Transformer dengan attention pada jendela lokal yang bergeser.

Perbandingan ini menilai konfigurasi eksperimen secara keseluruhan. Perbedaan hasil tidak dapat langsung dianggap sebagai pengaruh satu perubahan arsitektur karena ukuran input, batch size, lama training, dan strategi optimasi juga berbeda.

## Dataset dan aturan pembagian

Sumber data: [Mpox Skin Lesion Dataset v2.0 di Kaggle](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data).

- Training menggunakan **Original Train + Augmented Train pada fold yang sama**.
- Validasi menggunakan **Original Valid** untuk memilih checkpoint dengan validation loss terendah.
- Pengujian menggunakan **Original Test** dari fold yang sesuai, setelah pemilihan checkpoint.
- Model dan optimizer diinisialisasi ulang untuk masing-masing dari lima fold. Split bawaan dataset dipertahankan; tidak membuat pembagian acak baru.
- Dataset, arsip dataset, dan bobot model tidak diunggah ke repositori.

### Karakteristik distribusi data pada setiap kelas

Hitungan diperoleh langsung dari folder lokal, bukan dari perkiraan. Dalam satu fold lengkap terdapat **755 citra original** pada gabungan Train, Valid, dan Test. Tabel menggunakan fold 1 sebagai acuan; kelima fold memiliki total per kelas yang sama. Angka lima fold tidak dijumlahkan sebagai jumlah citra unik karena fold merupakan pembagian ulang koleksi yang sama.

| Kelas | Citra original dalam satu fold lengkap | Proporsi |
| --- | --- | --- |
| Chickenpox | 75 | 9.93% |
| Cowpox | 66 | 8.74% |
| HFMD | 161 | 21.32% |
| Healthy | 114 | 15.10% |
| Measles | 55 | 7.28% |
| Monkeypox | 284 | 37.62% |

Monkeypox merupakan kelas terbesar, sedangkan Measles paling sedikit. Datanya tidak seimbang, jumlah file Augmented Train pada setiap kelas adalah 14 kali jumlah Original Train, sehingga distribusi proporsi kelas tetap tidak seimbang. Jumlah file juga bukan merupakan jumlah pasien.

![Distribusi kelas](results/class_distribution.png)

| Fold | Original Train | Augmented Train | Train gabungan | Valid | Test |
| --- | --- | --- | --- | --- | --- |
| 1 | 537 | 7518 | 8055 | 144 | 74 |
| 2 | 509 | 7126 | 7635 | 164 | 82 |
| 3 | 538 | 7532 | 8070 | 153 | 64 |
| 4 | 524 | 7336 | 7860 | 150 | 81 |
| 5 | 538 | 7532 | 8070 | 154 | 63 |

Tabel lengkap per kelas dan split tersedia di [dataset_distribution.csv](results/dataset_distribution.csv). Penjumlahan Original dan Augmented Train mengikuti ketentuan tugas; transformasi augmentasi spesifik tidak direkonstruksi karena proyek menggunakan file augmentasi yang disediakan dataset.

### Preprocessing yang digunakan dan pengaruhnya terhadap hasil klasifikasi

Loader membaca citra dengan OpenCV, melakukan resize menjadi **100 × 100**, membagi intensitas dengan **255**, mengubah susunan kanal menjadi **CHW**, dan mengonversinya ke tensor `float32`. Urutan warna tetap **BGR**, sesuai kode dan checkpoint awal; tidak ada konversi RGB atau normalisasi ImageNet. CNN modifikasi dan Swin kemudian memperbesar tensor dengan interpolasi bilinear di dalam `forward()`.

Resize menyeragamkan dimensi input dan pembagian 255 menyetarakan skala intensitas. Namun, detail lesi yang hilang ketika diperkecil ke 100 × 100 tidak kembali hanya dengan diperbesar ke 160 atau 224. Tidak ada eksperimen ablation dengan model dan pengaturan identik, sehingga besar pengaruh preprocessing terhadap skor belum dapat diukur secara terpisah. Peningkatan CNN modifikasi tidak boleh diklaim semata-mata akibat resize, ELU, atau dropout.

### Kemampuan model untuk mengklasifikasi 6 kelas

**Belum secara merata.** CNN baseline memperoleh accuracy rata-rata **43,47%** dan CNN modifikasi **51,81%**. Recall per kelas menunjukkan kemampuan lebih baik pada HFMD dan Monkeypox daripada Chickenpox, Cowpox, dan Measles. Skor weighted dipengaruhi jumlah sampel kelas; karena itu confusion matrix dan recall setiap kelas juga diperiksa.

| Kelas | Recall rata-rata CNN | Recall rata-rata CNN modifikasi |
| --- | --- | --- |
| Chickenpox | 0.00% | 2.22% |
| Cowpox | 26.43% | 2.86% |
| HFMD | 51.08% | 73.08% |
| Healthy | 39.27% | 37.29% |
| Measles | 0.00% | 8.00% |
| Monkeypox | 70.61% | 84.88% |

Recall di tabel adalah rerata lima recall per fold, bukan recall dari satu confusion matrix gabungan. Swin belum dapat dinilai sebelum pengujian lima fold selesai.

### Kelas yang paling sering mengalami kesalahan klasifikasi

Pada CNN baseline, **Chickenpox dan Measles memiliki tingkat kesalahan rata-rata 100%** (recall 0%). Chickenpox paling sering diprediksi sebagai Monkeypox, dengan proporsi rata-rata 82,22% dari Chickenpox di masing-masing fold. Measles juga paling sering menjadi Monkeypox (66,33%). Jika memakai jumlah kemunculan pada seluruh Test fold, Chickenpox memiliki 42 kesalahan dari 42 kemunculan, paling banyak pada baseline.

Pada CNN modifikasi, tingkat kesalahan rata-rata tertinggi adalah **Chickenpox 97,78%**, diikuti **Cowpox 97,14%** dan **Measles 92,00%**. Chickenpox paling sering menjadi Monkeypox (78,43%); Cowpox juga paling sering menjadi Monkeypox (66,10%). Chickenpox salah pada 41 dari 42 kemunculan Test. Kemunculan antar fold tidak diperlakukan sebagai pasien atau citra unik.

Bukti numerik: [per kelas CNN](results/initial/cnn/per_class_summary.csv), [per kelas CNN modifikasi](results/initial/cnn_modified/per_class_summary.csv), serta `confusion_counts_fold*.csv` di folder model masing-masing.

![Confusion matrix CNN](results/initial/cnn/confusion_matrix_mean.png)

![Confusion matrix CNN modifikasi](results/initial/cnn_modified/confusion_matrix_mean.png)

### Hasil evaluasi pada masing-masing fold

| Model | Fold | Accuracy | Precision weighted | Recall weighted | F1 weighted | ROC-AUC macro OvR |
| --- | --- | --- | --- | --- | --- | --- |
| CNN baseline | 1 | 52.70% | 41.00% | 52.70% | 43.32% | 0.7791 |
| CNN baseline | 2 | 46.34% | 36.47% | 46.34% | 39.10% | 0.7105 |
| CNN baseline | 3 | 46.88% | 39.43% | 46.88% | 41.04% | 0.6952 |
| CNN baseline | 4 | 44.44% | 32.94% | 44.44% | 35.56% | 0.7788 |
| CNN baseline | 5 | 26.98% | 28.02% | 26.98% | 22.85% | 0.7406 |
| CNN modifikasi | 1 | 44.59% | 26.60% | 44.59% | 33.32% | 0.7394 |
| CNN modifikasi | 2 | 56.10% | 44.74% | 56.10% | 49.43% | 0.7819 |
| CNN modifikasi | 3 | 57.81% | 51.90% | 57.81% | 53.46% | 0.7359 |
| CNN modifikasi | 4 | 48.15% | 46.37% | 48.15% | 41.32% | 0.7339 |
| CNN modifikasi | 5 | 52.38% | 47.84% | 52.38% | 43.98% | 0.8124 |

CNN baseline memiliki accuracy tertinggi pada fold 1 (52,70%) dan terendah pada fold 5 (26,98%). CNN modifikasi tertinggi pada fold 3 (57,81%) dan terendah pada fold 1 (44,59%). Variasi hasil menandakan performa sensitif terhadap pembagian data. Ukuran Test bervariasi dari 63 sampai 82 gambar per fold.

CSV asli: [CNN baseline](results/initial/cnn/metrics_5fold.csv) dan [CNN modifikasi](results/initial/cnn_modified/metrics_5fold.csv). Angka tersebut berhasil direproduksi saat evaluasi ulang seluruh checkpoint dengan selisih maksimum 0.0 terhadap CSV awal. Swin masih menunggu hasil.

### Hasil rata-rata evaluasi dari kelima fold

#### Rata-rata lima fold

| Model | Accuracy | Precision weighted | Recall weighted | F1 weighted | ROC-AUC macro OvR |
| --- | --- | --- | --- | --- | --- |
| CNN baseline | 43,47 ± 9,72% | 35,57 ± 5,22% | 43,47 ± 9,72% | 36,37 ± 8,08% | 0,7408 ± 0,0384 |
| CNN modifikasi | 51,81 ± 5,48% | 43,49 ± 9,81% | 51,81 ± 5,48% | 44,30 ± 7,74% | 0,7607 ± 0,0350 |
| RegNetY-400MF | 75,85 ± 5,80% | 76,97 ± 6,60% | 75,85 ± 5,80% | 75,11 ± 5,90% | 0,9335 ± 0,0110 |

Nilai ditampilkan sebagai **mean ± standar deviasi sampel** (`ddof=1`) dari lima fold. RegNet unggul **24,04 poin
persentase** pada accuracy dan **30,81 poin persentase** pada F1 weighted dari CNN modifikasi. Keunggulan ini tidak
dapat dikaitkan hanya pada arsitektur, karena RegNet juga memakai bobot pretrained ImageNet sedangkan kedua CNN dilatih
dari nol.

#### Hasil per fold

| Model | Fold | Accuracy | Precision weighted | Recall weighted | F1 weighted | ROC-AUC macro OvR |
| --- | --- | --- | --- | --- | --- | --- |
| CNN baseline | 1 | 52,70% | 41,00% | 52,70% | 43,32% | 0,7791 |
| CNN baseline | 2 | 46,34% | 36,47% | 46,34% | 39,10% | 0,7105 |
| CNN baseline | 3 | 46,88% | 39,43% | 46,88% | 41,04% | 0,6952 |
| CNN baseline | 4 | 44,44% | 32,94% | 44,44% | 35,56% | 0,7788 |
| CNN baseline | 5 | 26,98% | 28,02% | 26,98% | 22,85% | 0,7406 |
| CNN modifikasi | 1 | 44,59% | 26,60% | 44,59% | 33,32% | 0,7394 |
| CNN modifikasi | 2 | 56,10% | 44,74% | 56,10% | 49,43% | 0,7819 |
| CNN modifikasi | 3 | 57,81% | 51,90% | 57,81% | 53,46% | 0,7359 |
| CNN modifikasi | 4 | 48,15% | 46,37% | 48,15% | 41,32% | 0,7339 |
| CNN modifikasi | 5 | 52,38% | 47,84% | 52,38% | 43,98% | 0,8124 |
| RegNetY-400MF | 1 | 74,32% | 76,72% | 74,32% | 73,55% | 0,9234 |
| RegNetY-400MF | 2 | 69,51% | 68,82% | 69,51% | 68,68% | 0,9207 |
| RegNetY-400MF | 3 | 81,25% | 81,95% | 81,25% | 80,82% | 0,9374 |
| RegNetY-400MF | 4 | 71,60% | 72,47% | 71,60% | 70,74% | 0,9396 |
| RegNetY-400MF | 5 | 82,54% | 84,87% | 82,54% | 81,74% | 0,9464 |

CNN baseline tertinggi pada fold 1 (52,70%) dan terendah pada fold 5 (26,98%). CNN modifikasi tertinggi pada fold 3
(57,81%) dan terendah pada fold 1 (44,59%). RegNet tertinggi pada fold 5 (82,54%) dan terendah pada fold 2 (69,51%).
Ukuran Test hanya 63 sampai 82 gambar per fold, sehingga skor cukup sensitif terhadap pembagian data.

CSV asli: [CNN baseline](results/initial/cnn/metrics_5fold.csv),
[CNN modifikasi](results/initial/cnn_modified/metrics_5fold.csv), dan
[RegNet](results/regnet/metrics_5fold.csv).

### Kelebihan dan keterbatasan model yang dikembangkan

| Model | Kelebihan | Keterbatasan |
| --- | --- | --- |
| CNN baseline | Arsitektur sederhana, hanya 50.876 parameter, menjadi acuan materi | Recall Chickenpox dan Measles 0%; training loss menurun ketika validation loss meningkat |
| CNN modifikasi | Accuracy dan F1 lebih tinggi dari baseline; BatchNorm, ELU, dropout, dan pooling global sudah diterapkan | Recall kelas minoritas tetap rendah; hasil awal belum berasal dari training Original + Augmented yang diperbaiki; beberapa komponen diubah sekaligus |
| RegNetY-400MF | Hasil terbaik pada semua metrik; bobot pretrained ImageNet membantu dataset kecil; arsitektur efisien (± 0,4 GFLOPs) | ± 3,9 juta parameter, jauh lebih berat dari dua CNN; input 100 × 100 jauh lebih kecil dari ukuran pretraining 224 × 224; perbandingan dengan CNN tidak setara karena inisialisasi berbeda |

Keterbatasan bersama meliputi ketidakseimbangan kelas, ukuran Test kecil, tidak adanya evaluasi pada dataset
eksternal, dan belum adanya analisis per pasien. Eksperimen ini mengevaluasi klasifikasi gambar untuk tugas perkuliahan
dan belum memvalidasi kemampuan diagnosis klinis.

## Grafik training

| CNN baseline | CNN modifikasi | RegNetY-400MF |
| --- | --- | --- |
| ![Training CNN fold 1](results/initial/cnn/training_fold1.png) | ![Training CNN modifikasi fold 1](results/initial/cnn_modified/training_fold1.png) | ![Training RegNet fold 1](results/regnet/training_fold1.png) |

Grafik baseline fold 1 menunjukkan indikasi overfitting; checkpoint yang dipakai untuk Test dipilih pada epoch 1, saat
validation loss terendah. CNN modifikasi fold 1 memakai checkpoint epoch 3. Daftar epoch terbaik kedua CNN terdapat di
[verification.json](results/initial/verification.json).

## Cara menjalankan

1. Clone repositori, buat virtual environment, lalu instal dependensi. Versi yang diuji tercantum pada `requirements.txt` (Python lokal 3.14.4). Untuk GPU, gunakan pasangan torch/torchvision yang sesuai petunjuk resmi PyTorch.

```powershell
git clone https://github.com/filzahghaniyyah/mpox-skin-lesion-classification.git
cd mpox-skin-lesion-classification
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Unduh dan ekstrak dataset Kaggle secara lokal. `--data-root` menunjuk folder induk berikut, bukan langsung ke `FOLDS`.

```text
DATA_ROOT/
├── Original Images/Original Images/FOLDS/
│   ├── fold1/Train|Valid|Test/<kelas>/
│   └── ... fold5/
└── Augmented Images/Augmented Images/FOLDS_AUG/
    ├── fold1_AUG/Train/<kelas>/
    └── ... fold5_AUG/
```

3. Jalankan training lalu testing untuk model yang diperlukan. Ganti path contoh dengan lokasi datasetmu. Semua perintah berjalan dari root repositori. `--device cpu` adalah default; gunakan `--device cuda` jika tersedia.

```powershell
.\.venv\Scripts\python.exe "Train (CNN).py" --data-root "D:\data\msld"
.\.venv\Scripts\python.exe "Test (CNN).py" --data-root "D:\data\msld"

.\.venv\Scripts\python.exe "Train (CNN modified).py" --data-root "D:\data\msld"
.\.venv\Scripts\python.exe "Test (CNN modified).py" --data-root "D:\data\msld"

.\.venv\Scripts\python.exe "Train (Swin).py" --data-root "D:\data\msld" --device cuda
.\.venv\Scripts\python.exe "Test (Swin).py" --data-root "D:\data\msld" --device cuda
```

`--epochs`, `--batch-size`, `--learning-rate`, dan `--seed` dapat diatur pada training. `--output-root` mengubah root keluaran. Testing juga menerima `--checkpoint-root` untuk mengevaluasi checkpoint yang sudah ada tanpa memindahkannya. Setiap root tersebut berisi subfolder `model 1 (CNN)`, `model 2 (CNN modified)`, atau `model 3 (Swin)`.

Checkpoint terbaik dan output baru disimpan lokal di `outputs/`. Folder ini diabaikan Git. Hasil training yang sedang berjalan tidak perlu dimulai ulang hanya untuk memakai repositori ini; setelah selesai, jalankan Test dengan checkpoint yang sesuai. Jangan mengubah preprocessing model di tengah satu eksperimen.

4. Setelah hasil lima fold lengkap, salin hanya tabel/grafik untuk dipublikasikan:

```powershell
.\.venv\Scripts\python.exe scripts/collect_results.py --model swin --run swin_5fold
.\.venv\Scripts\python.exe scripts/check_repository.py
```

Untuk hasil di folder lain, tambahkan `--source "path\ke\outputs"`. Script menolak hasil yang belum lengkap dan tidak menyalin dataset atau bobot. Perbarui tabel, jawaban LMS, status Swin, dan konfigurasi di README berdasarkan hasil baru sebelum melakukan commit. Panduan lanjutan: [melengkapi hasil](docs/next_steps.md).

## Struktur repositori

```text
Models/                Arsitektur tiga model
Utils/                 Loader, opsi eksekusi, dan ekspor per kelas
Train (*.py)           Training terpisah untuk setiap model
Test (*.py)            Evaluasi lima fold untuk setiap model
scripts/               Pengumpulan hasil dan pemeriksaan isi repositori
results/initial/       Grafik/CSV eksperimen awal dua CNN
results/               Distribusi dataset dan grafik perbandingan
docs/                  Riwayat eksperimen dan langkah melengkapi hasil
requirements.txt       Versi dependensi yang digunakan saat verifikasi
```

## Sumber dan atribusi

- Dokumen tugas W10 Studi Kasus Pembelajaran Mesin dan tujuh pertanyaan LMS menjadi dasar struktur laporan.
- CNN baseline mengikuti kode materi yang digunakan dalam proyek kelompok. CNN modifikasi diadaptasi dari notebook Rafa lalu disesuaikan menjadi tiga blok sebagaimana source lokal terakhir.
- [Dataset MSLD v2.0](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data).
- [Swin Transformer, Liu dkk.](https://arxiv.org/abs/2103.14030) dan [implementasi resmi torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html).
- [Dokumentasi ROC-AUC scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

Ketentuan tugas mengecualikan model yang sudah digunakan dalam kode proyek Kaggle yang dirujuk. Daftar implementasi pembanding Kaggle dan daftar algoritma materi belum diverifikasi lengkap di repositori ini; pemilihan Swin sebagai model ketiga perlu dicocokkan dengan batasan tersebut sebelum pengumpulan final.
