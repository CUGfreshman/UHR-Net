import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50
from .ughr import UGHRBlock


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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(attn)


class DilatedConvBranch(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(ConvBNReLU(in_c, out_c, kernel_size=1, padding=0), ChannelAttention(out_c))
        self.c2 = nn.Sequential(ConvBNReLU(in_c, out_c, kernel_size=3, padding=6, dilation=6), ChannelAttention(out_c))
        self.c3 = nn.Sequential(ConvBNReLU(in_c, out_c, kernel_size=3, padding=12, dilation=12), ChannelAttention(out_c))
        self.c4 = nn.Sequential(ConvBNReLU(in_c, out_c, kernel_size=3, padding=18, dilation=18), ChannelAttention(out_c))
        self.c5 = ConvBNReLU(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = ConvBNReLU(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = SpatialAttention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        merged = self.c5(torch.cat([x1, x2, x3, x4], dim=1))
        shortcut = self.c6(x)
        out = self.relu(merged + shortcut)
        return self.sa(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        self.relu = nn.ReLU(inplace=True)

        self.c1 = ConvBNReLU(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = ConvBNReLU(out_c, out_c, act=False)
        self.c3 = ConvBNReLU(out_c, out_c, act=False)
        self.c4 = ConvBNReLU(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        return self.sa(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_c, out_c=1):
        super().__init__()
        self.up_2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.fuse = ConvBNReLU(in_c * 3, in_c, kernel_size=3, padding=1)
        self.c1 = ConvBNReLU(in_c, 128, kernel_size=3, padding=1)
        self.c2 = ConvBNReLU(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f2, f3):
        f2 = self.up_2x(f2)
        f3 = self.up_4x(f3)

        fused = torch.cat([f1, f2, f3], dim=1)
        fused = self.fuse(fused)

        fused = self.up_2x(fused)
        fused = self.c1(fused)
        fused = self.c2(fused)
        fused = self.c3(fused)
        return self.sigmoid(fused)


class UHRNet(nn.Module):
    def __init__(self, use_ughr=True, num_prototypes=8, num_heads=8, logit_scale=1.0):
        super().__init__()
        self.use_ughr = use_ughr

        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.proj1 = ConvBNReLU(64, 256, kernel_size=3, padding=1)
        self.proj2 = ConvBNReLU(256, 256, kernel_size=3, padding=1)
        self.proj3 = ConvBNReLU(512, 256, kernel_size=3, padding=1)
        self.proj4 = ConvBNReLU(1024, 256, kernel_size=3, padding=1)

        # Coarse probability map M_hat for entropy-guided refinement.
        self.coarse_head = nn.Sequential(
            ConvBNReLU(1024, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.to_ughr4 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_local4 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_ughr3 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_local3 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_ughr2 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_local2 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_ughr1 = ConvBNReLU(256, 128, kernel_size=1, padding=0)
        self.to_local1 = ConvBNReLU(256, 128, kernel_size=1, padding=0)

        self.local_refine1 = DilatedConvBranch(128, 128)
        self.local_refine2 = DilatedConvBranch(128, 128)
        self.local_refine3 = DilatedConvBranch(128, 128)
        self.local_refine4 = DilatedConvBranch(128, 128)

        self.ughr1 = UGHRBlock(128, num_fg_prototypes=num_prototypes, num_bg_prototypes=num_prototypes, num_heads=num_heads, logit_scale=logit_scale)
        self.ughr2 = UGHRBlock(128, num_fg_prototypes=num_prototypes, num_bg_prototypes=num_prototypes, num_heads=num_heads, logit_scale=logit_scale)
        self.ughr3 = UGHRBlock(128, num_fg_prototypes=num_prototypes, num_bg_prototypes=num_prototypes, num_heads=num_heads, logit_scale=logit_scale)
        self.ughr4 = UGHRBlock(128, num_fg_prototypes=num_prototypes, num_bg_prototypes=num_prototypes, num_heads=num_heads, logit_scale=logit_scale)

        self.fusion_gate1 = nn.Sequential(
            ConvBNReLU(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.fusion_gate2 = nn.Sequential(
            ConvBNReLU(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.fusion_gate3 = nn.Sequential(
            ConvBNReLU(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.fusion_gate4 = nn.Sequential(
            ConvBNReLU(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        self.lateral_fuse3 = ConvBNReLU(128 + 256, 256, kernel_size=3, padding=1)
        self.lateral_fuse2 = ConvBNReLU(128 + 256, 256, kernel_size=3, padding=1)
        self.lateral_fuse1 = ConvBNReLU(128 + 256, 256, kernel_size=3, padding=1)

        self.decoder_low = DecoderBlock(128, 128, scale=2)
        self.decoder_mid = DecoderBlock(128, 128, scale=2)
        self.decoder_high = DecoderBlock(128, 128, scale=2)

        self.segmentation_head = SegmentationHead(128, 1)

    def forward(self, image):
        x1 = self.layer0(image)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        feat1 = self.proj1(x1)
        feat2 = self.proj2(x2)
        feat3 = self.proj3(x3)
        feat4 = self.proj4(x4)

        h1, w1 = feat1.shape[2:]
        feat2_up = F.interpolate(feat2, size=(h1, w1), mode="bilinear", align_corners=False)
        feat3_up = F.interpolate(feat3, size=(h1, w1), mode="bilinear", align_corners=False)
        feat4_up = F.interpolate(feat4, size=(h1, w1), mode="bilinear", align_corners=False)
        coarse_prob = self.coarse_head(torch.cat([feat1, feat2_up, feat3_up, feat4_up], dim=1))

        if self.use_ughr:
            bsz, _, h4, w4 = feat4.shape
            ughr_in4 = self.to_ughr4(feat4)
            local_in4 = self.to_local4(feat4)
            ughr_seq4 = ughr_in4.flatten(2).transpose(1, 2)
            coarse_4 = F.interpolate(coarse_prob, size=(h4, w4), mode="bilinear", align_corners=False)
            ughr_out4 = self.ughr4(ughr_seq4, coarse_4).transpose(1, 2).view(bsz, 128, h4, w4)
            local_out4 = self.local_refine4(local_in4)
            gate4 = self.fusion_gate4(torch.cat([ughr_out4, local_out4], dim=1))
            ref4 = gate4 * ughr_out4 + (1.0 - gate4) * local_out4
        else:
            ref4 = self.local_refine4(self.to_local4(feat4))

        bsz, _, h3, w3 = feat3.shape
        feat3_fuse = torch.cat([feat3, F.interpolate(ref4, size=(h3, w3), mode="bilinear", align_corners=False)], dim=1)
        feat3_fuse = self.lateral_fuse3(feat3_fuse)

        if self.use_ughr:
            ughr_in3 = self.to_ughr3(feat3_fuse)
            local_in3 = self.to_local3(feat3_fuse)
            ughr_seq3 = ughr_in3.flatten(2).transpose(1, 2)
            coarse_3 = F.interpolate(coarse_prob, size=(h3, w3), mode="bilinear", align_corners=False)
            ughr_out3 = self.ughr3(ughr_seq3, coarse_3).transpose(1, 2).view(bsz, 128, h3, w3)
            local_out3 = self.local_refine3(local_in3)
            gate3 = self.fusion_gate3(torch.cat([ughr_out3, local_out3], dim=1))
            ref3 = gate3 * ughr_out3 + (1.0 - gate3) * local_out3
        else:
            ref3 = self.local_refine3(self.to_local3(feat3_fuse))

        bsz, _, h2, w2 = feat2.shape
        feat2_fuse = torch.cat([feat2, F.interpolate(ref3, size=(h2, w2), mode="bilinear", align_corners=False)], dim=1)
        feat2_fuse = self.lateral_fuse2(feat2_fuse)

        if self.use_ughr:
            ughr_in2 = self.to_ughr2(feat2_fuse)
            local_in2 = self.to_local2(feat2_fuse)
            ughr_seq2 = ughr_in2.flatten(2).transpose(1, 2)
            coarse_2 = F.interpolate(coarse_prob, size=(h2, w2), mode="bilinear", align_corners=False)
            ughr_out2 = self.ughr2(ughr_seq2, coarse_2).transpose(1, 2).view(bsz, 128, h2, w2)
            local_out2 = self.local_refine2(local_in2)
            gate2 = self.fusion_gate2(torch.cat([ughr_out2, local_out2], dim=1))
            ref2 = gate2 * ughr_out2 + (1.0 - gate2) * local_out2
        else:
            ref2 = self.local_refine2(self.to_local2(feat2_fuse))

        bsz, _, h1, w1 = feat1.shape
        feat1_fuse = torch.cat([feat1, F.interpolate(ref2, size=(h1, w1), mode="bilinear", align_corners=False)], dim=1)
        feat1_fuse = self.lateral_fuse1(feat1_fuse)

        if self.use_ughr:
            ughr_in1 = self.to_ughr1(feat1_fuse)
            local_in1 = self.to_local1(feat1_fuse)
            ughr_seq1 = ughr_in1.flatten(2).transpose(1, 2)
            ughr_out1 = self.ughr1(ughr_seq1, coarse_prob).transpose(1, 2).view(bsz, 128, h1, w1)
            local_out1 = self.local_refine1(local_in1)
            gate1 = self.fusion_gate1(torch.cat([ughr_out1, local_out1], dim=1))
            ref1 = gate1 * ughr_out1 + (1.0 - gate1) * local_out1
        else:
            ref1 = self.local_refine1(self.to_local1(feat1_fuse))

        low_feat = self.decoder_low(ref2, ref1)
        mid_feat = self.decoder_mid(ref3, ref2)
        high_feat = self.decoder_high(ref4, ref3)

        refined_prob = self.segmentation_head(low_feat, mid_feat, high_feat)
        return refined_prob, coarse_prob
