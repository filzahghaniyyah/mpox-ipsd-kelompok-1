"""Ekspor bukti per kelas untuk pertanyaan LMS tentang kesalahan prediksi."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def export_class_details(output_dir, fold, classes, truth, prediction):
    labels = list(range(len(classes)))
    counts = confusion_matrix(truth, prediction, labels=labels)
    pd.DataFrame(counts, index=classes, columns=classes).to_csv(
        output_dir / f"confusion_counts_fold{fold}.csv", index_label="actual"
    )
    report = classification_report(truth, prediction, labels=labels,
                                   target_names=classes, output_dict=True, zero_division=0)
    rows = []
    for i, name in enumerate(classes):
        row = {"class": name, **report[name]}
        row["misclassified"] = int(counts[i].sum() - counts[i, i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / f"per_class_fold{fold}.csv", index=False)


def summarize_classes(output_dir, classes):
    counts = np.stack([
        pd.read_csv(output_dir / f"confusion_counts_fold{fold}.csv", index_col=0).to_numpy()
        for fold in range(1, 6)
    ])
    normalized = counts / counts.sum(axis=2, keepdims=True)
    mean = normalized.mean(axis=0)
    rows = []
    for i, name in enumerate(classes):
        wrong = mean[i].copy()
        wrong[i] = -1
        destination = int(wrong.argmax())
        rows.append({
            "class": name, "recall_mean": mean[i, i], "error_rate_mean": 1 - mean[i, i],
            "test_appearances": int(counts[:, i, :].sum()),
            "misclassified_appearances": int(counts[:, i, :].sum() - counts[:, i, i].sum()),
            "most_common_wrong_prediction": classes[destination] if wrong[destination] > 0 else "none",
            "wrong_prediction_rate_mean": max(0, wrong[destination]),
        })
    pd.DataFrame(rows).to_csv(output_dir / "per_class_summary.csv", index=False)
