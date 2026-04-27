import torch
import torch.nn as nn
from torchvision import models


class CelebA_CNN(nn.Module):
    def __init__(self, num_attributes=40):
        super(CelebA_CNN, self).__init__()
        # Using ResNet18 as a pre-trained feature extractor
        self.backbone = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        num_ftrs = self.backbone.fc.in_features

        # Remove the final fully connected layer of ResNet
        self.backbone.fc = nn.Identity()

        # Head 1: Attribute Classification (40 binary traits)
        self.attr_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attributes)
        )

        # Head 2: Landmark Regression (10 coordinates)
        # Head 2: Landmark Regression (10 coordinates)
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