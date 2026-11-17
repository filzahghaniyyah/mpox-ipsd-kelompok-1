import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN
from pathlib import Path


from Utils.runtime import options, select_device, save_run_config, save_history


def main():
    args = options("cnn", training=True)
    BATCH_SIZE = args.batch_size
    EPOCH = args.epochs
    LEARNING_RATE = args.learning_rate

    device = select_device(args.device)
    root = args.data_root
    output_dir = args.output_dir
    save_run_config(args, "cnn")
    original_dataset_path = root / "Original Images" / "Original Images" / "FOLDS"
    augmented_dataset_path = root / "Augmented Images" / "Augmented Images" / "FOLDS_AUG"

    for fold in range(1, 6):
        torch.manual_seed(args.seed)  # Set the random seed for reproducibility
        original_train = getImageLabel(original_dataset_path / f"fold{fold}" / "Train")
        augmented_train = getImageLabel(augmented_dataset_path / f"fold{fold}_AUG" / "Train")
        valid_dataset = getImageLabel(original_dataset_path / f"fold{fold}" / "Valid")
        train_dataset = ConcatDataset([original_train, augmented_train])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = SimpleCNN(input_dim=100, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu').to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_valid_all = [], []
        best_valid_loss = float("inf")
        for epoch in range(EPOCH):
            train_loss = 0
            valid_loss = 0
            valid_correct = 0

            model.train()
            for batch, (src, trg) in enumerate(train_loader):
                src, trg = src.to(device), trg.to(device)
                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.item() * src.size(0)  # Multiply by batch size to get total loss for the batch

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
            model.eval()
            with torch.no_grad():
                for batch, (src, trg) in enumerate(valid_loader):
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'valid_loss': valid_loss,
                    'fold': fold,
                    'training_data': 'Original Train + Augmented Train',
                    'classes': tuple(original_train.classes),
                }, output_dir / f"SimpleCNN_fold{fold}_best.pt")

        save_history(output_dir, fold, loss_train_all, loss_valid_all)
        plt.figure()
        plt.plot(range(1, EPOCH + 1), loss_train_all, color="#ffc1e2", label='Training')
        plt.plot(range(1, EPOCH + 1), loss_valid_all, color="#bde3ff", label='Validation')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SimpleCNN - fold {fold}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_fold{fold}.png")
        plt.close()


if __name__ == "__main__":
    main()