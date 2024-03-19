import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(UNet, self).__init__()
        # ------------------------ Laboratoire 2 - Question 5 - Début de la section à compléter ------------------------
        self.hidden = None

        # Down 1
        self.conv_1_1 = None
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = None
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = None
        self.conv_2_1 = None
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = None
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = None
        self.conv_3_1 = None
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = None
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = None
        self.conv_4_1 = None
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = None
        self.relu_4_2 = nn.ReLU()

        # Down 5
        self.maxpool_5 = None
        self.conv_5_1 = None
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = None
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = None
        self.conv_6_1 = None
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = None
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = None
        self.conv_7_1 = None
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = None
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = None
        self.conv_8_1 = None
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = None
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = None
        self.conv_9_1 = None
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = None
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # Down 1
        # To do
        down1 = nn.Sequential(self.conv_1_1, self.relu_1_1, self.conv_1_2, self.relu_1_2)

        # Down 2
        # To do
        down2 = nn.Sequential(self.maxpool_2, self.conv_2_1, self.relu_2_1, self.conv_2_2, self.relu_2_2)

        # Down 3
        # To do
        down3 = nn.Sequential(self.maxpool_3, self.conv_3_1, self.relu_3_1, self.conv_3_2, self.relu_3_2)

        # Down 4
        # To do
        down4 = nn.Sequential(self.maxpool_4, self.conv_4_1, self.relu_4_1, self.conv_4_2, self.relu_4_2)

        # Down 5
        # To do
        down5 = nn.Sequential(self.maxpool_5, self.conv_5_1, self.relu_5_1, self.conv_5_2, self.relu_5_2)

        # Up 6
        # To do
        up6 = nn.Sequential(self.upsample_6, self.conv_6_1, self.relu_6_1, self.conv_6_2, self.relu_6_2)

        # Up 7
        # To do
        up7 = nn.Sequential(self.upsample_7, self.conv_7_1, self.relu_7_1, self.conv_7_2, self.relu_7_2)    

        # Up 8
        # To do
        up8 = nn.Sequential(self.upsample_8, self.conv_8_1, self.relu_8_1, self.conv_8_2, self.relu_8_2)        
        
        # Up 9
        # To do
        up9 = nn.Sequential(self.upsample_9, self.conv_9_1, self.relu_9_1, self.conv_9_2, self.relu_9_2)    
        
        # Out
        out = nn.Sequential(down1, down2, down3, down4, down5, up6, up7, up8, up9, self.output_conv)

        return out
        # ------------------------ Laboratoire 2 - Question 5 - Fin de la section à compléter --------------------------
