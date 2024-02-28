import torch
import torch.nn as nn
from metrics import detection_intersection_over_union

class ModelObjectDetection(nn.Module):
     def __init__(self):
        super(ModelObjectDetection, self).__init__()
        self.model = nn.Sequential(     
                    nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    nn.Dropout(0.2),

                    nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.2),

                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),

                    nn.Flatten(),

                    nn.ReLU(),
                    nn.Linear(1152,48),
                    nn.ReLU(),
                    nn.Linear(48,48),
                    nn.ReLU(),
                    nn.Linear(48,21),
                    nn.Sigmoid())
        # )
                                

     def forward(self, input):
            output = self.model(input)
            
            return torch.stack((output[:,0:7],output[:,7:14],output[:,14:21]),dim=1)

class LossObjectDetection(nn.Module):
    def __init__(self):
        super(LossObjectDetection, self).__init__()

        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()
        self.CrossLoss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        
        boxLoss = 0
        objectLoss = 0
        noObjectLoss = 0
        classLoss = 0

        for pred, tar in zip(output, target):
             for pred_box, tar_box in zip(pred, tar):
                if tar_box[0] == 1:
                    x_diff = (tar_box[1] - pred_box[1])**2
                    y_diff = (tar_box[2] - pred_box[2])**2
                    hw_diff = (torch.sqrt(tar_box[3]) - torch.sqrt(pred_box[3]))**2
                    boxLoss += x_diff + y_diff + 2*hw_diff

                    targetIoU = detection_intersection_over_union(pred_box[1:4], tar_box[1:4])
                    objectLoss += self.BCELoss(pred_box[0], targetIoU)

                    targetOH = nn.functional.one_hot(tar_box[4].long(), 3)
                    classLoss += self.CrossLoss(pred_box[4:], targetOH.float())
                else:
                    noObjectLoss += self.BCELoss(pred_box[0], tar_box[0])
   
        loss = 5*boxLoss + 2*classLoss + objectLoss + 1.5*noObjectLoss

        return loss