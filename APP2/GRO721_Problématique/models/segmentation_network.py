import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.hidden = 25
        input_channels = 1
        n_classes = 4

        # Down 1
        self.conv_1_1 =nn.Conv2d(input_channels,self.hidden,(3,3), (1,), 1)# nn.Conv2d(input_channels, self.hidden, (3,3), (1,), 1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(self.hidden, self.hidden, (3,3), (1,), 1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_2_1 = nn.Conv2d(self.hidden, self.hidden*2, (3,3), (1,), 1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(self.hidden*2, self.hidden*2, (3,3), (1,), 1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_3_1 = nn.Conv2d(self.hidden*2, self.hidden*4, (3,3), (1,), 1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(self.hidden*4, self.hidden*4, (3,3), (1,), 1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, padding=0, stride=(2,))
        self.conv_4_1 = nn.Conv2d(self.hidden*4, self.hidden*8, (3,3), (1,), 1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(self.hidden*8, self.hidden*4, (3,3), (1,), 1)
        self.relu_4_2 = nn.ReLU()

        # Up 5
        self.upsample_5 = nn.ConvTranspose2d(self.hidden*4, self.hidden*4, (3, 3), (2,), (0,))
        self.conv_5_1 = nn.Conv2d(self.hidden*8, self.hidden*4, (3,3), (1,), 1)
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(self.hidden*4, self.hidden*2, (3,3), (1,), 1)
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(self.hidden*2, self.hidden*2, (2, 2), (2,), (0,))
        self.conv_6_1 = nn.Conv2d(self.hidden*4, self.hidden*2, (3,3), (1,), 1)
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(self.hidden*2, self.hidden*1, (3,3), (1,), 1)
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(self.hidden*1, self.hidden*1, (3, 3), (2,), (0,))
        self.conv_7_1 = nn.Conv2d(self.hidden*2, self.hidden*1, (3,3), (1,), 1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(self.hidden*1, self.hidden*1, (3,3), (1,), 1)
        self.relu_7_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # Down 1
        down1 = self.relu_1_2(self.conv_1_2(self.relu_1_1(self.conv_1_1(x))))

        # Down 2
        down2 = self.relu_2_2(self.conv_2_2(self.relu_2_1(self.conv_2_1(self.maxpool_2(down1)))))

        # Down 3
        down3 = self.relu_3_2(self.conv_3_2(self.relu_3_1(self.conv_3_1(self.maxpool_3(down2)))))

        # Down 4
        down4 = self.relu_4_2(self.conv_4_2(self.relu_4_1(self.conv_4_1(self.maxpool_4(down3)))))

        # Up 5
        #temp_upsample5 = self.upsample_5(down4)
        concat5 = torch.concat((self.upsample_5(down4), down3), 1)
        up5 = self.relu_5_2(self.conv_5_2(self.relu_5_1(self.conv_5_1(concat5))))

        # Up 6
        concat6 = torch.concat((self.upsample_6(up5), down2), 1)
        up6 = self.relu_6_2(self.conv_6_2(self.relu_6_1(self.conv_6_1(concat6))))

        # Up 7
        concat7 = torch.concat((self.upsample_7(up6), down1), 1)
        up7 = self.relu_7_2(self.conv_7_2(self.relu_7_1(self.conv_7_1(concat7))))

        # Out
        out = self.output_conv(up7)

        return out




def getSegmentationCriterion():
    return nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1,0.1], dtype=torch.float32))