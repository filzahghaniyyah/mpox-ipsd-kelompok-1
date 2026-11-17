"""Tolak file dataset, checkpoint, rahasia, atau file besar sebelum publikasi."""
import subprocess
from pathlib import Path

BLOCKED_SUFFIXES = {".pt", ".pth", ".h5", ".hdf5", ".keras", ".caffe", ".caffemodel", ".onnx", ".ckpt", ".safetensors", ".pkl", ".pickle", ".bin", ".zip", ".rar", ".7z"}
BLOCKED_DIRS = {"original images", "augmented images", "folds", "folds_aug", "data", "dataset", "datasets", ".venv", "venv"}


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, stderr=subprocess.DEVNULL).decode().split("\0")
        files = [root / p for p in files if p]
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    problems = []
    for p in files:
        relative = p.relative_to(root)
        if (p.suffix.lower() in BLOCKED_SUFFIXES or
                any(part.lower() in BLOCKED_DIRS for part in relative.parts) or
                p.name.startswith('.env') or p.stat().st_size > 10_000_000):
            problems.append(str(relative))
    if problems:
        raise SystemExit("Jangan upload file berikut:\n" + "\n".join(problems))
    print(f"OK: {len(files)} file diperiksa; tidak ada dataset/checkpoint/file besar.")


if __name__ == "__main__":
    main()
