import torch
import torch.nn as nn
from torchvision.models import swin_t

class SwinModel(nn.Module):
    def __init__(self, input_dim=100, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu'):
        super(SwinModel, self).__init__()

        self.model = swin_t(weights = None)  # Load the Swin Transformer model without pre-trained weights
        self.model.head = nn.Linear(self.model.head.in_features, output)  # Replace the classification head with a new one for our specific output classes


    def forward(self, x):
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
        return self.model(x)