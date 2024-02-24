import torch
import torch.nn as nn
import torchvision

class ModelObjectDetection(nn.Module):
     def __init__(self):
        super(ModelObjectDetection, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=11),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(7744, 3500),
            nn.ReLU(),
            nn.Linear(3500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 21))  #3 boites par images avec 7 caracteristiques par boite
                                  #3 valeurs de classification, 3 valeurs de position + 1 valeurs de presence d'object

        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=1)
    
     def forward(self, x):
            output = self.model(x)
            box1Presence = self.sigmoid(output[:, 0:1])
            box1Class = self.softmax(output[:, 1:4])
            box1Coord = self.sigmoid(output[:, 4:7])
            box1 = torch.concat((box1Presence, box1Coord, box1Class), dim=1)

            box2Presence = self.sigmoid(output[:, 7:8])
            box2Class = self.softmax(output[:, 8:11])
            box2Coord = self.sigmoid(output[:, 11:14])
            box2 = torch.concat((box2Presence, box2Coord, box2Class), dim=1)

            box3Presence = self.sigmoid(output[:, 14:15])
            box3Class = self.softmax(output[:, 15:18])
            box3Coord = self.sigmoid(output[:, 18:21])
            box3 = torch.concat((box3Presence, box3Coord, box3Class), dim=1)

            outputStack = torch.stack((box1, box2, box3), dim=1)

            return outputStack

class LossObjectDetection(nn.Module):
    def __init__(self):
        super(LossObjectDetection, self).__init__()

        self.MSELoss = nn.MSELoss(reduction="sum")
        self.BCELoss = nn.BCELoss(reduction="sum")
        self.CrossLoss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, output, target):
        
        boxLoss = self.MSELoss(output[:,:,1:4], target[:,:,1:4])

        objectLoss = self.BCELoss(output[:,:,0], target[:,:,0].float())

        targetOH = nn.functional.one_hot(target[:,:,4].long(), 3)
        classLoss = self.CrossLoss(output[:,:,4:], targetOH.float())

        loss = 1.75*boxLoss + 1.75*classLoss + objectLoss

        return loss