from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Utils.getData import getImageLabel
from Models.CNN_modified import ModifiedCNN


def predict(model, data_loader, device):
    model.eval()
    y_true_list, y_prob_list = [], []
    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(device)
            pred = model(src)
            probs = torch.softmax(pred, dim=1)
            y_true_list.extend(trg.argmax(dim=1).cpu().tolist())
            y_prob_list.extend(probs.cpu().tolist())
    return torch.tensor(y_true_list), torch.tensor(y_prob_list)


def main():
    BATCH_SIZE = 32
    EPOCH = 30
    LEARNING_RATE = 0.001
    MODEL_NAME = "ModifiedCNN"
    N_CLASSES = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs" / "model 2 (CNN modified)"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    augmented_dataset_path = root / "Augmented Images" / "Augmented Images" / "FOLDS_AUG"

    for fold in range(1, 6):
        torch.manual_seed(42)
        original_train = getImageLabel(original_dataset_path / f"fold{fold}" / "Train")
        augmented_train = getImageLabel(augmented_dataset_path / f"fold{fold}_AUG" / "Train")
        train_dataset = ConcatDataset([original_train, augmented_train])

        valid_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Valid")
        test_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Test")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ModifiedCNN(output=6, dropout1=0.4, dropout2=0.3).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_valid_loss = float("inf")

        for epoch in range(EPOCH):
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for src, trg in train_loader:
                src, trg = src.to(device), trg.to(device)
                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.item() * src.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(device), trg.to(device)
                    pred = model(src)
                    loss = loss_function(pred, trg)
                    valid_loss += loss.item() * src.size(0)

            train_loss /= len(train_loader.dataset)
            valid_loss /= len(valid_loader.dataset)
            scheduler.step(valid_loss)

            loss_train_all.append(train_loss)
            loss_valid_all.append(valid_loss)
            print(
                "fold =", fold,
                "epoch =", epoch + 1,
                "train loss =", train_loss,
                "valid loss =", valid_loss
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "valid_loss": valid_loss,
                    "fold": fold,
                    "classes": tuple(original_train.classes),
                }, output_dir / f"ModifiedCNN_fold{fold}_best.pt")

        checkpoint = torch.load(output_dir / f"ModifiedCNN_fold{fold}_best.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        y_true, probs = predict(model, test_loader, device)
        y_pred = probs.argmax(1)
        p, r, f1, _ = precision_recall_fscore_support(y_true.numpy(), y_pred.numpy(), average="macro", zero_division=0)

        row = {
            "model": MODEL_NAME, "fold": fold,
            "accuracy": accuracy_score(y_true.numpy(), y_pred.numpy()),
            "precision_macro": p, "recall_macro": r, "f1_macro": f1,
            "roc_auc_score": roc_auc_score(y_true.numpy(), probs.numpy(), multi_class="ovr", average="macro", labels=list(range(N_CLASSES))),
            "best_epoch": checkpoint["epoch"] + 1,
        }
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()})

        plt.figure()
        plt.plot(range(1, EPOCH + 1), loss_train_all, color="#ffc1e2", label='Training')
        plt.plot(range(1, EPOCH + 1), loss_valid_all, color="#bde3ff", label='Validation')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_fold{fold}.png")
        plt.close()


if __name__ == "__main__":
    main()
