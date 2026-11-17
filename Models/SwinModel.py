import torch
import torch.nn as nn
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights


class RegNetModel(nn.Module):
    def __init__(self, output=6):
        super(RegNetModel, self).__init__()
        
        self.model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, output)

        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x[:, [2, 1, 0]]
        x = (x - self.mean) / self.std
        return self.model(x)
