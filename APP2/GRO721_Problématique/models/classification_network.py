import torch
import torch.nn as nn
import torchvision


def getClassificationModel():
    # alexnet = torchvision.models.alexnet(pretrained=True)
    
    # alexnet.features[0] = nn.Conv2d(1,64, kernel_size=8, stride=2, padding=11)
    # alexnet.features[3] = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
    # alexnet.features[6] = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    # alexnet.features[8] = nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1)
    # alexnet.features[10] = nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1)
    # alexnet.classifier[4] = nn.Linear(100, 50)
    # alexnet.classifier[6] = nn.Linear(50, 10)

    model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=11),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(400, 75),
            nn.ReLU(),
            nn.Linear(75, 25),
            nn.ReLU(),
            nn.Linear(25, 3),  #nous avons 3 classes de sortie
            nn.Sigmoid()
            )

    return model

def getClassificationCriterion():
    return nn.BCELoss()