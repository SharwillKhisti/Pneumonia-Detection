import torch.nn as nn
from torchvision import models

def get_densenet121(num_classes=2):
    model = models.densenet121(weights="DEFAULT")
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model