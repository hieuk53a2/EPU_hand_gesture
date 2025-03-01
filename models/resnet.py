import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet(num_classes, pretrained=True):
    resnet = models.resnet50(pretrained=pretrained)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    return resnet