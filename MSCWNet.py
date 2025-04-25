import torch
import torch.nn.functional as F
from MSCAM import MSCAM
from WTConv import WTConv
import torch.nn as nn
import math
from DySample import DySample


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            WTConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out





class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Sequential(
            DySample(in_channels // 2),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        )

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 对x1进行填充以匹配x2的尺寸
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 定义输出卷积层
class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MSCWNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, filters=None):
        super(MSCWNet, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        features = [64, 128, 256, 512]
        self.dca_model = MSCAM(features=features)
        self.filters = filters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, self.filters[0])


        self.down1 = Down(self.filters[0], self.filters[1])
        self.down2 = Down(self.filters[1], self.filters[2])
        self.down3 = Down(self.filters[2], self.filters[3])
        self.down4 = Down(self.filters[3], self.filters[3])



        self.up1 = Up(self.filters[4], self.filters[2])
        self.up2 = Up(self.filters[3], self.filters[1])
        self.up3 = Up(self.filters[2], self.filters[0])
        self.up4 = Up(self.filters[1], self.filters[0])



        self.outc = OutConv(self.filters[0], n_classes)  # 64 -> n_classes

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)  # [64, H, W]
        x2 = self.down1(x1)  # [128, H/2, W/2]
        x3 = self.down2(x2)  # [256, H/4, W/4]
        x4 = self.down3(x3)  # [512, H/8, W/8]
        x5 = self.down4(x4)  # [512, H/8, W/8]
        x1, x2, x3, x4 = self.dca_model((x1, x2, x3, x4))

        x6 = self.up1(x5, x4)  # [512, H/8, W/8]
        x7 = self.up2(x6, x3)  # [256, H/4, W/4]
        x8 = self.up3(x7, x2)  # [128, H/2, W/2]
        x9 = self.up4(x8, x1)  # [64, H, W]


        logits = self.outc(x9)  # [n_classes, H, W]
        return logits


