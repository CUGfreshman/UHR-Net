import torch
import torch.nn as nn
from .resnet import resnet50


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class UOICHead(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4, out_c):
        super().__init__()
        self.conv1 = ConvBNReLU(in_c1, 64, kernel_size=1, padding=0, act=True)
        self.conv2 = ConvBNReLU(in_c2, 64, kernel_size=1, padding=0, act=True)
        self.conv3 = ConvBNReLU(in_c3, 64, kernel_size=1, padding=0, act=True)
        self.conv4 = ConvBNReLU(in_c4, 64, kernel_size=1, padding=0, act=True)
        self.fuse = nn.Conv2d(4 * 64, out_c, kernel_size=1, padding=0, bias=False)

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)
        uoic_feat = torch.cat([f1, f2, f3, f4], dim=1)
        logits = self.fuse(uoic_feat)
        return logits, uoic_feat


class UOICPretrainNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.up_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up_8x = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        self.up_16x = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False)

        self.head = UOICHead(64, 256, 512, 1024, 1)

    def forward(self, image):
        x1 = self.layer0(image)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        x1 = self.up_2x(x1)
        x2 = self.up_4x(x2)
        x3 = self.up_8x(x3)
        x4 = self.up_16x(x4)

        # UO-IC uses the fused feature map for instance contrast.
        mask_logits, uoic_feat = self.head(x1, x2, x3, x4)
        return mask_logits, uoic_feat
