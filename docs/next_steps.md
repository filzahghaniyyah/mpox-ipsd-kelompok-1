# Menindaklanjuti hasil lima fold

## Hasil yang sudah tersedia

README memuat evaluasi lima fold CNN baseline, CNN modifikasi, dan RegNetY-400MF. Folder setiap model menyimpan `metrics_5fold.csv`, lima grafik training, lima confusion matrix, dan matriks rata-rata. [Tabel per fold](../README.md#hasil-evaluasi-pada-masing-masing-fold) · [Folder hasil](../results/initial)

## Pelatihan ulang CNN modifikasi

Untuk mengukur pengaruh protokol training gabungan, jalankan kembali `Train (CNN modified).py` lalu `Test (CNN modified).py` memakai dataset yang sama. Skrip training saat ini menggabungkan Original Train dan Augmented Train pada setiap fold. Simpan hasil dengan nama run tersendiri agar perbandingan dengan eksperimen awal tetap dapat ditelusuri. [Panduan menjalankan kode](../GUIDE.md) · [Riwayat eksperimen](experiment_history.md)

Jika output training dan testing berada di folder `outputs` bawaan, jalankan perintah berikut dari root repositori.

```powershell
python scripts/collect_results.py --source outputs --model cnn_modified --run cnn_modified_combined
```

Folder `results/cnn_modified_combined/cnn_modified/` akan memuat grafik dan CSV dari run tersebut. Saat memperbarui laporan, gunakan metrik Test, matriks per kelas, dan konfigurasi dari run yang sama.

## Pelaporan

Masukkan URL repositori pada bagian Metode di template laporan. README menyajikan tujuh pembahasan LMS, termasuk perbedaan hasil antarfold dan kelas yang paling sering salah. [README](../README.md)
