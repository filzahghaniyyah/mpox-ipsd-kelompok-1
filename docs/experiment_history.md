# Riwayat eksperimen dan verifikasi

Snapshot ini disiapkan pada 17 November 2026 berdasarkan source lokal dan hasil eksperimen yang sudah tersedia. Dataset dan checkpoint hanya dibaca lokal, tidak disertakan dalam repositori.

## Bukti yang tersedia

- Dua model CNN memiliki lima checkpoint, lima grafik training, lima confusion matrix, confusion matrix rata-rata, dan CSV metrik.
- Seluruh checkpoint dua CNN dievaluasi ulang dengan loader yang sama (BGR, 100×100, /255). Lima nilai per metrik cocok persis dengan CSV asli; maksimum selisih absolut 0.0.
- File `metrics_5fold.csv` dan semua PNG pada `results/initial/` dipertahankan dari hasil asli. CSV per kelas dan confusion counts ditambahkan dari evaluasi ulang.
- Swin sedang training menurut penulis. Uji satu batch forward/backward berhasil, tetapi ini bukan hasil training atau bukti akurasi.

## Perbedaan hasil awal dan kode yang dibagikan

| Bagian | Source / artefak awal | Versi repositori |
|---|---|---|
| Output baseline | Train/Test mencari checkpoint di root, hasil lokal sudah dipindah ke folder model | Selalu menggunakan subfolder model yang sama |
| Epoch baseline | Source tersimpan menunjukkan 100; grafik hasil awal menunjukkan 10 epoch | Default 10, dapat diatur `--epochs`; tidak mengklaim source 100 epoch menghasilkan grafik lama |
| Training CNN modifikasi | Source yang menyertai hasil hanya mengambil Augmented Train | Original Train + Augmented Train, sesuai tugas; hasil baru memerlukan training ulang |
| Evaluasi CNN modifikasi | Train juga membaca Test dan mencetak metrik setelah pemilihan checkpoint | Test hanya dibaca melalui script Test yang terpisah |
| Impor model modifikasi | Pengecekan dummy otomatis berjalan setiap import | Pengecekan dihapus dari import, arsitektur/state_dict tetap sama |
| Swin | Impor tidak terpakai; runtime tanpa konfigurasi CLI | Impor dirapikan, path/device dapat diatur; tetap Swin-Tiny dari nol |
| Metadata | Tidak menyimpan seluruh konfigurasi atau riwayat numerik | Run baru menyimpan run_config.json, history CSV, dan metadata sumber Train |

Checkpoint lama tidak menyimpan provenance split yang lengkap. Keterangan sumber data eksperimen awal didasarkan pada source yang menyertainya dan percakapan pengerjaan, bukan audit riwayat proses training. Karena itu hasil lama tidak dilabeli sebagai hasil dari kode training baru.

Perbandingan awal CNN baseline vs CNN modifikasi bersifat deskriptif. Selain arsitektur, strategi stopping, scheduler, input internal, dan source Train berbeda. Belum dilakukan ablation terkontrol atau pengujian signifikansi.

## Pemeriksaan yang dilakukan

- Parse sintaks semua berkas Python.
- Sampel data nyata: bentuk input CHW 3×100×100, rentang [0,1], label one-hot enam kelas.
- Forward dan backward satu batch untuk ketiga model: output 1×6 dan loss/gradien finite.
- Alur training lima fold untuk dua CNN dijalankan dengan subset enam citra per split dan satu epoch untuk memeriksa checkpoint, riwayat, serta grafik. Ini hanya uji integrasi; semua output sementara dibuang dan tidak digunakan sebagai hasil eksperimen.
- Evaluasi sepuluh checkpoint CNN pada seluruh Test fold serta pembandingan CSV.
- Pemeriksaan bahwa paket publik hanya berisi source, dokumentasi, grafik hasil, dan tabel agregat.

Lingkungan verifikasi: Python 3.14.4 dengan versi paket pada requirements.txt, CPU, dua thread PyTorch. Pelatihan penuh lima fold tidak dijalankan ulang pada tahap pengemasan repositori ini.
