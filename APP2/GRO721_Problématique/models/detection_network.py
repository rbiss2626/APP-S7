# import torch
# import torch.nn as nn
# import torchvision

# class ModelObjectDetection(nn.Module):
#      def __init__(self):
#         super(ModelObjectDetection, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             # nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.1), #16x27x27
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, 3, padding=1),
#             # nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1), #32x13x13
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), #64x6x6
#             nn.Conv2d(64, 64, 3, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), #64x6x6
#             nn.Conv2d(64, 64, 3, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), #64x6x6
#             nn.Conv2d(64, 64, 3, padding=1),
#             # nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), #64x6x6
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), #128x3x3
#             nn.Conv2d(128, 128, 3, padding=1),
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), #128x3x3
#             nn.Conv2d(128, 128, 3, padding=1),
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), #128x3x3
#             nn.Conv2d(128, 128, 3, padding=0, stride=1),
#             # nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), #128x1x1
#             nn.Conv2d(128, 21, 1),
#             # nn.BatchNorm2d(21),
#             nn.LeakyReLU(0.1) #21x1x1
#             # nn.LeakyReLU(0.1), #128x3x3
#             # nn.Conv2d(128, 128, 3, padding=1),
#             # nn.BatchNorm2d(128),
#             # nn.LeakyReLU(0.1), #128x3x3
#             # nn.MaxPool2d(2, 2),
#             # nn.Conv2d(128, 256, 3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.LeakyReLU(0.1), #256x1x1
#             # nn.Conv2d(256, 256, 3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.LeakyReLU(0.1), #256x1x1
#             # nn.Conv2d(256, 256, 3, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.LeakyReLU(0.1), #256x1x1
#             # nn.Conv2d(256, 21, 3, padding=1),
#             # nn.BatchNorm2d(21),
#             # nn.LeakyReLU(0.1) #21x1x1
#             )

#         self.sigmoid = nn.Sigmoid() 
    
#      def forward(self, x):
#         output = self.model(x)
#         output = self.sigmoid(output)
#         # output = output.flatten()
#         output = output.reshape(len(x), 3, 7)
#         return output

# class LossObjectDetection(nn.Module):
#     def __init__(self):
#         super(LossObjectDetection, self).__init__()

#         self.MSELoss = nn.MSELoss()
#         self.BCELoss = nn.BCELoss()
#         self.CrossLoss = nn.CrossEntropyLoss()

#     def forward(self, output, target):
        
#         # boxLoss = self.MSELoss(output[:,:,1:4], target[:,:,1:4])
#         boxLoss = self.MSELoss(output[:,:,1], target[:,:,1]) + self.MSELoss(output[:,:,2], target[:,:,2]) + 2*self.MSELoss(torch.sqrt(output[:,:,3]), torch.sqrt(target[:,:,3]))#(output[:,:,1] - target[:,:,1])**2 + (output[:,:,2] - target[:,:,2])**2 + (torch.sqrt(output[:,:,3]) - torch.sqrt(target[:,:,3]))**2 + (torch.sqrt(output[:,:,3]) - torch.sqrt(target[:,:,3]))**2

#         targetObj = (target[:,:,0] == 1).float()
#         targetNoObj = (target[:,:,0] == 0).float()

#         # IoU = torch.box_iou(output[:,:,1:4], target[:,:,1:4])
        
#         objectLoss = self.BCELoss(output[:,:,0], targetObj)
#         noObjectLoss = self.BCELoss(output[:,:,0], targetNoObj)

#         targetOH = nn.functional.one_hot(target[:,:,4].long(), 3)
#         classLoss = self.BCELoss(output[:,:,4:], targetOH.float())

#         loss = 5*boxLoss + classLoss + objectLoss + 0.5*noObjectLoss

#         return loss

import torch
import torch.nn as nn
import torchvision
from metrics import detection_intersection_over_union
import torch.nn.functional as F

class ModelObjectDetection(nn.Module):
     def __init__(self):
        super(ModelObjectDetection, self).__init__()
        self.model = nn.Sequential(     
                    nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    nn.Dropout(0.2),

                    nn.Conv2d(32, 96, kernel_size=5, stride=2, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(0.2),

                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    # nn.Dropout(0.2),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    # nn.Dropout(0.2),
                    # nn.MaxPool2d(kernel_size=2, stride=2),

                    nn.Flatten(),

                    nn.ReLU(),
                    nn.Linear(1152,48),
                    nn.ReLU(),
                    nn.Linear(48,48),
                    nn.ReLU(),
                    nn.Linear(48,21),
                    # nn.ReLU(),
                    # nn.Linear(750,500),
                    # nn.ReLU(),
                    # nn.Linear(500,250),
                    # nn.ReLU(),
                    # nn.Linear(250,21),
                    nn.Sigmoid())
        # )
                                 

    #     self.sigmoid = nn.Sigmoid()
    #     self.softmax = nn.Softmax()

     def forward(self, input):
            output = self.model(input)
            
            return torch.stack((output[:,0:7],output[:,7:14],output[:,14:21]),dim=1)
    #     self.model = nn.Sequential(
    #         nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=1),

    #         nn.Conv2d(32, 96, kernel_size=5, stride=2, padding=2),
    #         nn.BatchNorm2d(96),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),

    #         nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
    #         nn.BatchNorm2d(96),
    #         nn.ReLU(),
    #         # nn.MaxPool2d(kernel_size=2, stride=2),

    #         nn.Flatten(),

    #         nn.Linear(3456, 1000),
    #         nn.ReLU(),
    #         nn.Linear(1000, 300),
    #         nn.ReLU(),
    #         nn.Linear(300, 21))  #3 boites par images avec 7 caracteristiques par boite
    #                               #3 valeurs de classification, 3 valeurs de position + 1 valeurs de presence d'object

    #     self.sigmoid = nn.Sigmoid() 
    #     self.softmax = nn.Softmax(dim=1)
    
    #  def forward(self, x):
    #         output = self.model(x)
    #         box1Presence = self.sigmoid(output[:, 0:1])
    #         box1Class = self.softmax(output[:, 1:4])
    #         box1Coord = self.sigmoid(output[:, 4:7])
    #         box1 = torch.concat((box1Presence, box1Coord, box1Class), dim=1)

    #         box2Presence = self.sigmoid(output[:, 7:8])
    #         box2Class = self.softmax(output[:, 8:11])
    #         box2Coord = self.sigmoid(output[:, 11:14])
    #         box2 = torch.concat((box2Presence, box2Coord, box2Class), dim=1)

    #         box3Presence = self.sigmoid(output[:, 14:15])
    #         box3Class = self.softmax(output[:, 15:18])
    #         box3Coord = self.sigmoid(output[:, 18:21])
    #         box3 = torch.concat((box3Presence, box3Coord, box3Class), dim=1)

    #         outputStack = torch.stack((box1, box2, box3), dim=1)

    #         return outputStack

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