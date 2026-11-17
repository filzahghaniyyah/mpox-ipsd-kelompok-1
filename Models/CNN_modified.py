import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedCNN(nn.Module):
    """M2 Modified CNN dari notebook Rafa."""

    def __init__(self, output=6, dropout1=0.4, dropout2=0.3):
        super().__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(128, 128)
        self.activation = nn.ELU()
        self.dropout2 = nn.Dropout(dropout2)
        self.fc_out = nn.Linear(128, output)

    def forward(self, x):
        # getData.py memberi input 100x100; Rafa memakai ukuran M2 160x160. 
        if x.shape[2:] != (160, 160):
            x = F.interpolate(
                x, size=(160, 160), mode="bilinear", align_corners=False
            )

        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)

        # Logits; Softmax dilakukan saat evaluasi untuk ROC-AUC.
        return self.fc_out(x)
