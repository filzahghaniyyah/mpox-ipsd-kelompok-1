import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=100, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu'):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        feature_dim = (input_dim - 4) // 3  # after convBlock1
        feature_dim = (feature_dim - 4) // 3  # after convBlock
        self.fc = nn.Linear(in_features=50 * feature_dim * feature_dim, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(torch.flatten(x, 1))

        return x