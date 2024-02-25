import torch
import torch.nn as nn
import torchvision

class ModelObjectDetection(nn.Module):
     def __init__(self):
        super(ModelObjectDetection, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1), #16x27x27
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1), #32x13x13
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.1), #64x6x6
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1), #64x6x6
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1), #64x6x6
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1), #64x6x6
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1), #128x3x3
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1), #128x3x3
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1), #128x3x3
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1), #128x3x3
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1), #256x1x1
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 21, 3, padding=1),
            nn.LeakyReLU(0.1)
            )

        self.sigmoid = nn.Sigmoid() 
    
     def forward(self, x):
        output = self.model(x)
        output = self.sigmoid(output)
        output = output.flatten()
        output = output.reshape(len(x), 3, 7)
        return output

class LossObjectDetection(nn.Module):
    def __init__(self):
        super(LossObjectDetection, self).__init__()

        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()
        self.CrossLoss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        
        boxLoss = self.MSELoss(output[:,:,1:4], target[:,:,1:4])

        objectLoss = self.BCELoss(output[:,:,0], target[:,:,0].float())

        targetOH = nn.functional.one_hot(target[:,:,4].long(), 3)
        classLoss = self.CrossLoss(output[:,:,4:], targetOH.float())

        loss = 1.75*boxLoss + 1.75*classLoss + objectLoss

        return loss