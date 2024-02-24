import torch
import torch.nn as nn
import torchvision

class ModelObjectDetection(nn.Module):
     def __init__(self):
        super(ModelObjectDetection, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=11)

        self.model = nn.Sequential(
            self.alexnet,
            nn.ReLU(),
            nn.Linear(1000, 21), #3 boites par images avec 7 caracteristiques par boite
                                 #3 valeurs de classification, 3 valeurs de position + 1 valeurs de presence d'object
            nn.Sigmoid()
        )
    
     def forward(self, x):
          output = self.model(x)
          outputStack = torch.stack((output[:, 0:7], output[:, 7:14], output[:, 14:21]), dim=1)

          return outputStack
     

class LossObjectDetection(nn.Module):
    def __init__(self):
        super(LossObjectDetection, self).__init__()

        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()
        self.CrossLoss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        # outputLoc = torch.stack((output[:,1:4], output[:,8:11], output[:,15:18]), dim=1)
        targetLoc = target[:,:,1:4]
        boxLoss = self.MSELoss(output[:,:,1:4], targetLoc)

        # outputObj = torch.stack((output[:,0], output[:,7], output[:,14]), dim=1)
        objectLoss = self.BCELoss(output[:,:,0], target[:,:,0])

        targetOH = nn.functional.one_hot(target[:,:,4].long(), 3)
        # outputClass = torch.stack((output[:, 4:7], output[:, 11:14], output[:, 18:21]), dim=1)
        classLoss = self.CrossLoss(output[:,:,4:], targetOH.float())

        loss = boxLoss + classLoss + objectLoss

        return loss
#BOX1
    #index 0 -> prob object
    #index 1 -> pos x
    #index 2 -> pos y
    #index 3 -> size
    #index 4 -> cercle
    #index 5 -> triangle
    #index 6 -> croix
#BOX2
    #index 7 -> prob object
    #index 8 -> pos x
    #index 9 -> pos y
    #index 10 -> size
    #index 11 -> cercle
    #index 12 -> triangle
    #index 13 -> croix
#BOX 3
    #index 14 -> prob object
    #index 15 -> pos x
    #index 16 -> pos y
    #index 17 -> size
    #index 18 -> cercle
    #index 19 -> triangle
    #index 20 -> croix
    