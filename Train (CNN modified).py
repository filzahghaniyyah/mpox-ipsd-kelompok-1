import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Utils.getData import getImageLabel
from Models.CNN_modified import ModifiedCNN


from Utils.runtime import options, select_device, save_run_config, save_history


def main():
    args = options("cnn_modified", training=True)
    BATCH_SIZE = args.batch_size
    EPOCH = args.epochs
    LEARNING_RATE = args.learning_rate
    PATIENCE = 5


    device = select_device(args.device)
    root = args.data_root
    output_dir = args.output_dir
    save_run_config(args, "cnn_modified")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    augmented_dataset_path = root / "Augmented Images" / "Augmented Images" / "FOLDS_AUG"

    for fold in range(1, 6):
        torch.manual_seed(args.seed)

        original_train = getImageLabel(original_dataset_path / f"fold{fold}" / "Train")
        augmented_train = getImageLabel(augmented_dataset_path / f"fold{fold}_AUG" / "Train")
        train_dataset = ConcatDataset([original_train, augmented_train])
        valid_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Valid")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ModifiedCNN(output=6, dropout1=0.4, dropout2=0.3).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_valid_loss = float("inf")
        epochs_without_improvement = 0

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
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "valid_loss": valid_loss,
                    "fold": fold,
                    "training_data": "Original Train + Augmented Train",
                    "classes": (
                        "Chickenpox", "Cowpox", "HFMD",
                        "Healthy", "Measles", "Monkeypox"
                    ),
                }, output_dir / f"ModifiedCNN_fold{fold}_best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print("Early stopping pada epoch", epoch + 1)
                    break

        save_history(output_dir, fold, loss_train_all, loss_valid_all)
        plt.figure()
        epochs_done = range(1, len(loss_train_all) + 1)
        plt.plot(epochs_done, loss_train_all, label="Training")
        plt.plot(epochs_done, loss_valid_all, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_fold{fold}.png")
        plt.close()


if __name__ == "__main__":
    main()