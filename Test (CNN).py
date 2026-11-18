import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from pathlib import Path

from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    BATCH_SIZE = 32
    root = Path(__file__).resolve().parent
    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    classes = ("Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox")
    fold_metrics, fold_confusion_matrices = [], []

    for fold in range(1, 6):
        test_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE, shuffle=False
        )

        model = SimpleCNN(input_dim=100, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu')

        checkpoint = torch.load(
            root / f"SimpleCNN_fold{fold}_best.pt",
            map_location='cpu', weights_only=True
        )
        if checkpoint['fold'] != fold or tuple(checkpoint['classes']) != classes:
            raise ValueError(f"Fold atau urutan kelas checkpoint fold {fold} tidak cocok.")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"\nFold {fold}, jumlah data Test: {len(test_dataset)}, "
            f"checkpoint epoch: {checkpoint['epoch'] + 1}"
        )

        prediction, ground_truth, probability = [], [], []

        model.eval()
        with torch.no_grad():
            model.eval()
            for batch, (src, trg) in enumerate(test_loader):
                # getData.py sudah menghasilkan [batch, 3, 100, 100].
                pred = model(src)

                prediction.extend(torch.argmax(pred, dim=1).cpu().numpy())
                ground_truth.extend(torch.argmax(trg, dim=1).cpu().numpy())
                # ROC-AUC multikelas membutuhkan probabilitas setiap kelas.
                probability.extend(torch.softmax(pred, dim=1).cpu().numpy())

        # Build confusion matrix untuk keenam kelas.
        cf_matrix = confusion_matrix(ground_truth, prediction, labels=range(len(classes)))
        
        print(cf_matrix)
        row_sum = cf_matrix.sum(axis=1, keepdims=True)
        cf_normalized = np.divide(
            cf_matrix, row_sum,
            out=np.zeros_like(cf_matrix, dtype=float),
            where=row_sum != 0
        )
        fold_confusion_matrices.append(cf_normalized)

        df_cm = pd.DataFrame(
            cf_normalized, index=classes, columns=classes
        )
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.xlabel("Prediksi")
        plt.ylabel("Label asli")
        plt.title(f"SimpleCNN - fold {fold} - normalisasi per baris")
        plt.tight_layout()
        plt.savefig(root / f"confusion_matrix_fold{fold}.png")
        plt.close()

        accuracy = accuracy_score(ground_truth, prediction)
        precision = precision_score(
            ground_truth, prediction, average='weighted', zero_division=0
        )
        recall = recall_score(
            ground_truth, prediction, average='weighted', zero_division=0
        )
        f1 = f1_score(
            ground_truth, prediction, average='weighted', zero_division=0
        )
        roc_auc = roc_auc_score(
            ground_truth, np.asarray(probability),
            multi_class='ovr', average='macro',
            labels=list(range(len(classes)))
        )

        print("accuracy score =", accuracy)
        print("precision score (weighted) =", precision)
        print("recall score (weighted) =", recall)
        print("f1 score (weighted) =", f1)
        print("ROC-AUC (macro OvR) =", roc_auc)

        fold_metrics.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "roc_auc_macro_ovr": roc_auc,
        })

    # Hitung mean dan std hanya dari lima hasil fold.
    results = pd.DataFrame(fold_metrics).set_index("fold")
    mean_metrics = results.mean(axis=0)
    std_metrics = results.std(axis=0, ddof=1)

    results.loc["mean"] = mean_metrics
    results.loc["std"] = std_metrics

    print("\nHasil tiap fold, rerata, dan standar deviasi:")
    print(results.round(4).to_string())
    results.to_csv(root / "metrics_5fold.csv")

    # Rerata confusion matrix yang sudah dinormalisasi per fold.
    mean_cf = np.mean(fold_confusion_matrices, axis=0)
    df_cm = pd.DataFrame(mean_cf, index=classes, columns=classes)

    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2f", vmin=0, vmax=1)
    plt.xlabel("Prediksi")
    plt.ylabel("Label asli")
    plt.title("SimpleCNN - rerata confusion matrix 5-fold")
    plt.tight_layout()
    plt.savefig(root / "confusion_matrix_mean.png")
    plt.close()


if __name__ == "__main__":
    main()
