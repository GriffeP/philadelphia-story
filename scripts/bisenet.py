"""
BiSeNet face parsing model definition.

Matches the 79999_iter.pth checkpoint from zllrunning/face-parsing.PyTorch,
which uses a ResNet18-only architecture (no separate SpatialPath).

Segmentation labels (19 classes, CelebAMask-HQ):
    0: background    1: skin       2: nose       3: eye_glass   4: l_eye
    5: r_eye         6: l_brow     7: r_brow     8: l_ear        9: r_ear
   10: mouth        11: u_lip     12: l_lip      13: hair        14: hat
   15: earring      16: necklace  17: neck       18: cloth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.conv_atten = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.bn_atten(self.conv_atten(atten))
        atten = torch.sigmoid(atten)
        return feat * atten


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, 1, 1, 0)
        self.conv1 = nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.relu(self.conv1(atten))
        atten = torch.sigmoid(self.conv2(atten))
        return feat + feat * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, 1, 1, 0)

    def forward(self, x):
        # ResNet18 stages
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        feat8 = self.resnet.layer2(self.resnet.layer1(x))  # 1/8, 128ch
        feat16 = self.resnet.layer3(feat8)                   # 1/16, 256ch
        feat32 = self.resnet.layer4(feat16)                  # 1/32, 512ch

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32.shape[2:], mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, feat16.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8.shape[2:], mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class BiSeNetOutput(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, n_classes, 1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        # FFM fuses feat8 (128ch) with feat16_up (128ch) = 256ch input
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat8, feat_cp16, feat_cp32 = self.cp(x)

        feat_fuse = self.ffm(feat8, feat_cp16)
        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)

        out16 = self.conv_out16(feat_cp16)
        out32 = self.conv_out32(feat_cp32)
        out16 = F.interpolate(out16, (h, w), mode="bilinear", align_corners=True)
        out32 = F.interpolate(out32, (h, w), mode="bilinear", align_corners=True)

        return out, out16, out32
