"""Opsi path bersama; data dan checkpoint tetap disimpan secara lokal."""
import argparse
import json
from pathlib import Path

MODEL_FOLDERS = {
    "cnn": "model 1 (CNN)",
    "cnn_modified": "model 2 (CNN modified)",
    "swin": "model 3 (Swin)",
}


def options(model, training=False):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"{'Train' if training else 'Test'} {model}, 5 fold")
    parser.add_argument("--data-root", type=Path, default=root,
                        help="Folder yang berisi Original Images dan Augmented Images")
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8 if model == "swin" else 32)
    if training:
        parser.add_argument("--epochs", type=int, default=30 if model == "cnn_modified" else 10)
        parser.add_argument("--learning-rate", type=float, default=0.001)
        parser.add_argument("--seed", type=int, default=42)
    else:
        parser.add_argument("--checkpoint-root", type=Path,
                            help="Root outputs checkpoint; default sama dengan --output-root")
    args = parser.parse_args()
    if args.batch_size < 1 or (training and (args.epochs < 1 or args.learning_rate <= 0)):
        parser.error("Batch size, epochs, dan learning rate harus positif.")
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_root.resolve() / MODEL_FOLDERS[model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not training:
        args.checkpoint_dir = (args.checkpoint_root or args.output_root).resolve() / MODEL_FOLDERS[model]
    return args


def select_device(name):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia; gunakan --device cpu.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def save_run_config(args, model):
    import importlib.metadata as metadata
    config = {
        "model": model, "folds": [1, 2, 3, 4, 5], "seed": args.seed,
        "max_epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "device": str(select_device(args.device)),
        "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "training_data": "Original Train + Augmented Train, fold yang sama",
        "checkpoint_selection": "minimum Original Valid loss",
        "preprocessing": "OpenCV BGR, resize 100x100, /255, CHW float32",
        "early_stopping_patience": 5 if model == "cnn_modified" else None,
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2)" if model == "cnn_modified" else None,
        "versions": {p: metadata.version(p) for p in ("torch", "torchvision", "numpy", "opencv-python")},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_history(output_dir, fold, train_losses, valid_losses):
    import csv
    with (output_dir / f"history_fold{fold}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        writer.writerows((i + 1, t, v) for i, (t, v) in enumerate(zip(train_losses, valid_losses)))
