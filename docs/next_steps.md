# Melengkapi eksperimen sebelum pengumpulan

## Swin yang sedang berjalan

Biarkan training yang sedang berjalan selesai. Siapkan lima checkpoint SwinModel_fold1_best.pt sampai SwinModel_fold5_best.pt serta lima grafik training dari run yang sama. Salin checkpoint ke folder lokal yang sesuai bila training dilakukan di Colab. Jangan commit checkpoint.

Jalankan Test (Swin).py dari root repositori dengan --data-root menuju dataset lokal dan --checkpoint-root menuju root yang berisi subfolder model 3 (Swin). Gunakan --output-root menuju root output yang berisi grafik training dari run tersebut. Script memeriksa keberadaan kelima checkpoint, metadata fold, dan urutan kelas.

Kemudian jalankan:

```powershell
python scripts/collect_results.py --source "path/ke/outputs" --model swin --run swin_5fold
```

Hasil akan berada di results/swin_5fold/swin. Perbarui tabel per-fold, tabel mean/std, grafik perbandingan, serta jawaban pertanyaan 3–7 di README. Tulis konfigurasi sebenarnya dari run yang selesai; jangan menganggap default repositori otomatis sama dengan run Colab.

## CNN modifikasi

Hasil awal belum mewakili koreksi Original Train + Augmented Train. Jalankan ulang Train (CNN modified).py dan Test (CNN modified).py untuk lima fold, lalu kumpulkan dengan nama run baru seperti cnn_modified_combined. Pertahankan results/initial sebagai riwayat, dan gunakan hasil terbaru yang sesuai protokol untuk kesimpulan akhir.

## Pemeriksaan sebelum commit

Pastikan lima fold lengkap, semua model menggunakan sumber dataset yang diperbolehkan, dan algoritma ketiga sesuai materi serta tidak melanggar pengecualian kode Kaggle. Tambahkan identitas kelompok sesuai kebutuhan laporan. Tidak ada nama/NIM anggota yang dibuat-buat di repositori ini.

```powershell
git add README.md docs results Models Utils scripts requirements.txt .gitignore
python scripts/check_repository.py
git diff --cached --stat
git commit -m "Complete five-fold experiment results"
git push
```

Masukkan URL repositori pada bagian Metode di template laporan. README sudah mengikuti tujuh pertanyaan LMS; PPT enam menit dan laporan Word merupakan keluaran terpisah.
