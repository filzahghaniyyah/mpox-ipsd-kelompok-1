import torch
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from Models.RegNet import RegNetModel
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def main():
    BATCH_SIZE = 16
    device = get_device()
    print("Device:", device)
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "model 3 (RegNet)"
    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    classes = ("Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox")
    fold_metrics = []
    fold_confusion_matrices = []

    for fold in range(1, 6):
        checkpoint_path = output_dir / f"RegNetModel_fold{fold}_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint belum ditemukan: {checkpoint_path}. "
                "Selesaikan Train (RegNet).py untuk kelima fold dahulu."
            )

    for fold in range(1, 6):
        test_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Test")
        if len(test_dataset) == 0:
            raise ValueError(f"Data Test fold {fold} kosong.")
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = RegNetModel(output=6)
        checkpoint = torch.load(
            output_dir / f"RegNetModel_fold{fold}_best.pt",
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["fold"] != fold:
            raise ValueError(f"Checkpoint fold {fold} tidak cocok.")
        if tuple(checkpoint["classes"]) != classes or tuple(test_dataset.classes) != classes:
            raise ValueError(f"Urutan kelas pada fold {fold} tidak cocok.")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        del checkpoint

        prediction, ground_truth, probability = [], [], []
        model.eval()
        with torch.no_grad():
            for src, trg in test_loader:
                src = src.to(device)
                pred = model(src)
                prediction.extend(pred.argmax(dim=1).cpu().numpy())
                ground_truth.extend(trg.argmax(dim=1).cpu().numpy())
                probability.extend(torch.softmax(pred, dim=1).cpu().numpy())

        cf_matrix = confusion_matrix(ground_truth, prediction, labels=range(len(classes)))

        cf_normalized = np.divide(
            cf_matrix,
            np.sum(cf_matrix, axis=1)[:, None],
            out=np.zeros_like(cf_matrix, dtype=float),
            where=np.sum(cf_matrix, axis=1)[:, None] != 0,
        )
        fold_confusion_matrices.append(cf_normalized)

        plt.figure(figsize=(12, 7))
        sn.heatmap(
            pd.DataFrame(cf_normalized, index=classes, columns=classes),
            annot=True, fmt=".2f", vmin=0, vmax=1,
        )
        plt.xlabel("Prediksi")
        plt.ylabel("Label asli")
        plt.title(f"Confusion matrix RegNet fold {fold}")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_fold{fold}.png")
        plt.close()

        accuracy = accuracy_score(ground_truth, prediction)
        precision = precision_score(ground_truth, prediction, average="weighted", zero_division=0)
        recall = recall_score(ground_truth, prediction, average="weighted", zero_division=0)
        f1 = f1_score(ground_truth, prediction, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(
            ground_truth,
            np.asarray(probability),
            multi_class="ovr",
            average="macro",
            labels=list(range(len(classes))),
        )

        print(f"\nFold {fold}")
        print("accuracy score =", accuracy)
        print("precision score =", precision)
        print("recall score =", recall)
        print("f1 score =", f1)
        print("ROC-AUC =", roc_auc)

        fold_metrics.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "roc_auc_macro_ovr": roc_auc,
        })

    results = pd.DataFrame(fold_metrics).set_index("fold")
    results.loc["mean"] = results.mean(axis=0)
    print("\nHasil RegNet dan rerata 5-fold:")
    print(results.round(4).to_string())
    results.to_csv(output_dir / "metrics_5fold.csv")

    mean_cf = np.mean(fold_confusion_matrices, axis=0)
    plt.figure(figsize=(12, 7))
    sn.heatmap(
        pd.DataFrame(mean_cf, index=classes, columns=classes),
        annot=True, fmt=".2f", vmin=0, vmax=1,
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Label asli")
    plt.title("Rerata confusion matrix RegNet")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_mean.png")
    plt.close()


if __name__ == "__main__":
    main()