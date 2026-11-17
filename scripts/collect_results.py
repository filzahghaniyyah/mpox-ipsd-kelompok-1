"""Salin hanya grafik dan tabel evaluasi lengkap; tidak menyalin bobot/dataset."""
import argparse
import csv
import json
import shutil
from pathlib import Path

FOLDERS = {"cnn": "model 1 (CNN)", "cnn_modified": "model 2 (CNN modified)", "swin": "model 3 (Swin)"}
METRICS = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_macro_ovr")


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "outputs", help="Root output training/testing")
    parser.add_argument("--model", choices=FOLDERS, required=True)
    parser.add_argument("--run", required=True, help="Nama run baru, contoh swin_5fold atau cnn_modified_combined")
    args = parser.parse_args()
    if not args.run or not all(c.isalnum() or c in "_-" for c in args.run):
        parser.error("Nama run hanya boleh huruf, angka, underscore, atau tanda minus.")
    source = args.source.resolve() / FOLDERS[args.model]
    destination = repo / "results" / args.run / args.model
    if destination.exists():
        parser.error("Folder run sudah ada. Gunakan nama baru agar hasil lama tetap tersimpan.")
    required = ["metrics_5fold.csv", "confusion_matrix_mean.png"]
    required += [f"{kind}_fold{fold}.png" for kind in ("training", "confusion_matrix") for fold in range(1, 6)]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        parser.error("Hasil lima fold belum lengkap: " + ", ".join(missing))
    with (source / "metrics_5fold.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {row["fold"]: row for row in csv.DictReader(f)}
    if not all(str(f) in rows for f in range(1, 6)):
        parser.error("CSV harus memuat kelima fold.")
    values = {m: [float(rows[str(f)][m]) for f in range(1, 6)] for m in METRICS}
    if not all(0 <= n <= 1 for series in values.values() for n in series):
        parser.error("Metrik harus berupa angka valid antara 0 dan 1.")
    # Hanya nama artefak evaluasi yang dikenal yang boleh disalin.
    optional = ["run_config.json", "per_class_summary.csv"]
    optional += [f"{kind}_fold{fold}.csv" for kind in ("history", "confusion_counts", "per_class") for fold in range(1, 6)]
    destination.mkdir(parents=True)
    for name in required + optional:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    summary = {m: sum(v) / 5 for m, v in values.items()}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"# Hasil {args.model}\n\nRerata lima fold, dihitung dari baris 1–5.\n\n| Metrik | Nilai |\n|---|---:|\n"
    report += "".join(f"| {m} | {value:.6f} |\n" for m, value in summary.items())
    report += "\n![Confusion matrix](confusion_matrix_mean.png)\n\nKonfigurasi hasil harus dicocokkan dengan run_config.json atau catatan eksperimen.\n"
    (destination / "README.md").write_text(report, encoding="utf-8")
    print(f"Hasil tersalin ke {destination}. Perbarui README utama sebelum commit.")


if __name__ == "__main__":
    main()
