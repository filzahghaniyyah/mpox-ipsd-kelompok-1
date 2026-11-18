import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.RegNet import RegNetModel
from pathlib import Path


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def to_cpu(obj):
    #buat pindahin semua tensor ke CPU sebelum disimpan ke checkpoint.
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    return obj


def main():
    BATCH_SIZE = 16
    EPOCH = 10
    LEARNING_RATE = 0.001
    device = get_device()
    print("Device:", device)

    root = Path(__file__).parent
    output_dir = root / "outputs" / "model 3 (RegNet)"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    augmented_dataset_path = root / "Augmented Images" / "Augmented Images" / "FOLDS_AUG"

    for fold in range(1, 6):
        torch.manual_seed(42)  # Set the random seed for reproducibility
        original_train = getImageLabel(original_dataset_path / f"fold{fold}" / "Train")
        augmented_train = getImageLabel(augmented_dataset_path / f"fold{fold}_AUG" / "Train")
        valid_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Valid")
        train_dataset = ConcatDataset([original_train, augmented_train])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = RegNetModel(output=6).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_valid_loss = float("inf")
        for epoch in range(EPOCH):
            train_loss = 0
            valid_loss = 0
            valid_correct = 0

            model.train()
            for batch_idx, (src, trg) in enumerate(train_loader, start=1):
                src, trg = src.to(device), trg.to(device)

                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.item() * src.size(0)  # Multiply by batch size to get total loss for the batch

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Progress supaya kelihatan programnya jalan
                if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                    print(f"  fold {fold} epoch {epoch + 1}: batch {batch_idx}/{len(train_loader)}", end="\r")
            print()

            model.eval()
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(device), trg.to(device)
                    pred = model(src)
                    loss = loss_function(pred, trg)
                    valid_loss += loss.item() * src.size(0)

                    valid_correct += (
                        torch.argmax(pred, dim=1)
                        == torch.argmax(trg, dim=1)
                    ).sum().item()

            # Rata-rata loss per gambar.
            train_loss /= len(train_dataset)
            valid_loss /= len(valid_dataset)
            valid_accuracy = valid_correct / len(valid_dataset)

            loss_train_all.append(train_loss)
            loss_valid_all.append(valid_loss)

            print(
                "fold =", fold,
                "epoch =", epoch + 1,
                "train loss =", train_loss,
                "valid loss =", valid_loss,
                "valid accuracy =", valid_accuracy
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': to_cpu(model.state_dict()),
                    'optimizer_state_dict': to_cpu(optimizer.state_dict()),
                    'loss': train_loss,
                    'valid_loss': valid_loss,
                    'fold': fold,
                    'classes': tuple(original_train.classes),
                }, output_dir / f"RegNetModel_fold{fold}_best.pt")

        plt.figure()
        plt.plot(range(1, EPOCH + 1), loss_train_all, color="#ffc1e2", label='Training')
        plt.plot(range(1, EPOCH + 1), loss_valid_all, color="#bde3ff", label='Validation')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"RegNetModel - fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_fold{fold}.png")
        plt.close()


if __name__ == "__main__":
    main()