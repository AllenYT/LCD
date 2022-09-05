from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

input_size = (256, 256)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel), #归一化
            nn.Dropout2d(0,3), #正则化
            nn.LeakyReLU(), #激活
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 归一化
            nn.Dropout2d(0, 3),  # 正则化
            nn.LeakyReLU() # 激活
            #layers.add
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer = nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2,mode='nearest')
        out = self.layer(up)
        #print("suc")
        return torch.cat((out,feature_map),dim=1)

class LCDNet(nn.Module):

    def __init__(self, num_classes=3, dropout_2d=0.2,
                 pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        # encoder
        self.encoder = torchvision.models.resnet50(pretrained=pretrained)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv_Block(3,64)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.conv6 = nn.Conv2d(256, 128, 3, 1, 1, padding_mode='reflect', bias=False)
        # classification decoder
        self.avgpool = self.encoder.avgpool
        self.fc = nn.Linear(512 * 4, self.num_classes)
        # segmentation decoder
        self.u1 = UpSample(2048)
        self.c6 = Conv_Block(1024+1024, 1024)
        self.u2 = UpSample(1024)
        self.c7 = Conv_Block(512+512, 512)
        self.u3 = UpSample(512)
        self.c8 = Conv_Block(256+256, 256)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(64+64, 64)
        self.out = nn.Conv2d(64,3,3,1,1)
        self.Th = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.encoder.maxpool(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        # classification
        x = self.avgpool(conv5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # segmentation
        y = self.c6(self.u1(conv5, conv4))              # 2048->1024
        y = self.c7(self.u2(y, conv3))                  # 1024->512
        y = self.c8(self.u3(y, conv2))                  # 512->256
        y = self.conv6(y)                               # 256->128
        y = self.c9(self.u4(y, conv1))                  # 128->64
        y = self.out(y)
        y = self.Th(y)
        return x,y


# torch.Size([4, 64, 256, 256])
# torch.Size([4, 256, 128, 128])
# torch.Size([4, 512, 64, 64])
# torch.Size([4, 1024, 32, 32])
# torch.Size([4, 2048, 16, 16])
# torch.Size([4, 2048, 1, 1])
# torch.Size([4, 2048])
# torch.Size([4, 3])
# torch.Size([4, 1024, 32, 32])
# torch.Size([4, 512, 64, 64])
# torch.Size([4, 256, 128, 128])
# torch.Size([4, 128, 128, 128])
# torch.Size([4, 64, 256, 256])
# torch.Size([4, 3, 256, 256])
# torch.Size([4, 3, 256, 256])