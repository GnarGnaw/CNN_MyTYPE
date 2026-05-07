import torch
import torch.nn as nn
from torchvision import models


class CelebA_CNN(nn.Module):
    def __init__(self, num_attributes=40):
        super(CelebA_CNN, self).__init__()
        self.backbone = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.attr_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attributes)
        )

        self.land_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        features = self.backbone(x)
        attributes = self.attr_head(features)
        landmarks = self.land_head(features)
        return attributes, landmarks