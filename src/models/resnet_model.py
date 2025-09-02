import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=2):
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50(num_classes=2):
    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model