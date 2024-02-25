import torch
import torch.nn as nn
import torchvision

class ModelSegmentation(nn.Module):
     def __init__(self):
        super(ModelSegmentation, self).__init__()

        #Down1        
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1),
        self.relu1_1 = nn.ReLU(inplace=True),
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1),
        self.relu1_2 = nn.ReLU(inplace=True),

        self.maxPool1_1 = nn.MaxPool2d(kernel_size=2, stride=2),

        #Down2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1),
        self.relu2_1 = nn.ReLU(inplace=True),
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1),
        self.relu2_2 = nn.ReLU(inplace=True),

        self.maxPool2_1 = nn.MaxPool2d(kernel_size=2, stride=2),

        #Down3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1),
        self.relu3_1 = nn.ReLU(inplace=True),
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1),
        self.relu3_2 = nn.ReLU(inplace=True),

        self.maxPool3_1 = nn.MaxPool2d(kernel_size=2, stride=2),

        #Down4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1),
        self.relu4_1 = nn.ReLU(inplace=True),
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1),
        self.relu4_2 = nn.ReLU(inplace=True),

        self.maxPool4_1 = nn.MaxPool2d(kernel_size=2, stride=2),

        #Down5
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1),
        self.relu5_1 = nn.ReLU(inplace=True),
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        self.relu5_2 = nn.ReLU(inplace=True),

        self.upSample1_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        self.concat1_1 = torch.concat(1, 512),

        #Up1
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        self.relu6_1 = nn.ReLU(inplace=True),
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1),
        self.relu6_2 = nn.ReLU(inplace=True),

        self.upSample1_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        self.concat1_2 = torch.concat(1, 256),

        #Up2
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1),
        self.relu7_1 = nn.ReLU(inplace=True),
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1),
        self.relu7_2 = nn.ReLU(inplace=True),

        self.upSample1_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        self.concat1_3 = torch.concat(1, 128),


        #Up3
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1),
        self.relu8_1 = nn.ReLU(inplace=True),
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1),
        self.relu8_2 = nn.ReLU(inplace=True),

        self.upSample1_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        self.concat1_4 = torch.concat(1, 64),

        #Up4
        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1),
        self.relu9_1 = nn.ReLU(inplace=True),
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1),
        self.relu9_2 = nn.ReLU(inplace=True),

        nn.Sigmoid()
        


     def forward(self, x):
          output = self.model(x)
          return output