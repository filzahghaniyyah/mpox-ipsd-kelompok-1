import cv2 as cv
import torch
import numpy as np
from os import listdir, path
from torch.utils.data import Dataset


class getImageLabel(Dataset):
    def __init__(self, folder):
        self.dataset = []
        self.classes = ("Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox")
        # initiate the one hot encoding for class
        self.to_one_hot = np.eye(6)  # number of class
        
        for i, class_name in enumerate(self.classes):
            class_folder = path.join(folder, class_name)

            if not path.isdir(class_folder):
                raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_folder}")
            for image_name in sorted(listdir(class_folder)):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = path.join(class_folder, image_name)
                    # append feature image and one hot encoded label to public self.dataset
                    self.dataset.append([image_path, i])

        if not self.dataset:
            raise ValueError(f"Dataset kosong: {folder}")

    def __getitem__(self, item):
        image_path, label = self.dataset[item]
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        # BGR dipertahankan agar konsisten dengan checkpoint eksperimen awal.
        image = cv.resize(image, (100, 100))/255.0
        image = np.transpose(image, (2, 0, 1))  # Change to channel-first format
        return (torch.tensor(image, dtype=torch.float32), torch.tensor(self.to_one_hot[label], dtype=torch.float32))
    
    def __len__(self):
        return len(self.dataset)