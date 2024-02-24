import torch
import torch.nn as nn
import torchvision


def getClassificationModel():
    alexnet = torchvision.models.alexnet(pretrained=True)
    
    alexnet.features[0] = nn.Conv2d(1,64, kernel_size=8, stride=2, padding=11)
    alexnet.features[3] = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
    alexnet.features[6] = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    alexnet.features[8] = nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1)
    alexnet.features[10] = nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1)

    alexnet.classifier[1] = nn.Linear(16*6*6, 100)
    alexnet.classifier[4] = nn.Linear(100, 50)
    alexnet.classifier[6] = nn.Linear(50, 10)

    model = nn.Sequential(
            alexnet,
            nn.ReLU(),
            nn.Linear(10, 3), #nous avons 3 classes de sortie
            nn.Sigmoid()
            )

    return model

def getClassificationCriterion():
    return nn.BCELoss()