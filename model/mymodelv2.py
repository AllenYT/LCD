import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torchvision import models

from model.unet_parts import *


class LCDNet(nn.Module):

    def __init__(self, cla_num_classes=3, seg_num_classes=2,pretrained=True,bilinear=True):
        
        super().__init__()

        self.cla_num_classes = cla_num_classes
        self.seg_num_classes = seg_num_classes
        
        # encoder
        self.encoder = models.resnet34(pretrained=pretrained)
        self.conv1 = DoubleConv(3,64)
        self.conv2 = self.encoder.layer1            # 64
        self.conv3 = self.encoder.layer2            # 128
        self.conv4 = self.encoder.layer3            # 256
        self.conv5 = self.encoder.layer4            # 512

        # classification decoder
        self.avgpool = self.encoder.avgpool
        self.fc = nn.Linear(512, self.cla_num_classes)
        
        # segmentation decoder
        self.up1 = Up(512+256, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.seg_num_classes)

    def forward(self, x):                           
        x1 = self.conv1(x)                          # 3->64
        x2 = self.conv2(self.encoder.maxpool(x1))   # 64->64
        x3 = self.conv3(x2)                         # 64->128
        x4 = self.conv4(x3)                         # 128->256
        x5 = self.conv5(x4)                         # 256->512

        # classification
        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # segmentation
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.outc(y)
        return x,y

