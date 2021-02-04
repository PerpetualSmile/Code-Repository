import torch
import torch.nn as nn
import torch.nn.functional as F
from resnest.torch import resnest50
import torchvision

class ResNest(nn.Module):
    def __init__(self):
        super().__init__()
#         self.model = torchvision.models.densenet161(pretrained=True)
#         self.model.classifier = nn.Linear(2208, num_class)   
        self.model = resnest50(pretrained=True)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(1024, 24)
        )
        
    def forward(self, X):
        return self.model(X)