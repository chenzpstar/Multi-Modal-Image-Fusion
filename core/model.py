# -*- coding: utf-8 -*-
"""
# @file name  : model.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模型类
"""

import torch
import torch.nn as nn

try:
    from .block import *
    from .fusion import *
except:
    from block import *
    from fusion import *

__all__ = [
    'DeepFuse', 'DenseFuse', 'VIFNet', 'DIFNet', 'DBNet', 'PFNetv1', 'PFNetv2'
]


class _FusionModel(nn.Module):
    '''Base class for siamese-style fusion models.'''
    def __init__(self):
        super(_FusionModel, self).__init__()
        self.encode = nn.Sequential()
        self.decode = nn.Sequential()

    def encoder(self, img):
        return self.encode(img)

    def fusion(self, feat1, feat2):
        raise NotImplementedError

    def decoder(self, feat):
        return self.decode(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feat = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img


class DeepFuse(_FusionModel):
    '''DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs'''
    def __init__(self):
        super(DeepFuse, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5, padding=2),
            ConvBlock(16, 32, kernel_size=7, padding=3),
        )
        self.decode = nn.Sequential(
            ConvBlock(32, 32, kernel_size=7, padding=3),
            ConvBlock(32, 16, kernel_size=5, padding=2),
            ConvBlock(16, 1, kernel_size=5, padding=2, relu=False),
        )

    def fusion(self, feat1, feat2, mode='add'):
        return element_fusion(feat1, feat2, mode)


class DenseFuse(_FusionModel):
    '''DenseFuse: A Fusion Approach to Infrared and Visible Images'''
    def __init__(self):
        super(DenseFuse, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.decode = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def fusion(self, feat1, feat2, mode='add'):
        if mode == 'add':
            return element_fusion(feat1, feat2, mode='add')
        elif mode == 'l1':
            return spatial_fusion(feat1, feat2, mode='l1', softmax=False)
        else:
            raise ValueError("only supported ['add', 'l1'] mode")


class VIFNet(_FusionModel):
    '''VIF-Net: An Unsupervised Framework for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(VIFNet, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.decode = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def fusion(self, feat1, feat2):
        return concat_fusion(feat1, feat2)


class DIFNet(_FusionModel):
    '''Unsupervised Deep Image Fusion With Structure Tensor Representations'''
    def __init__(self):
        super(DIFNet, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.fuse = ConvBlock(32, 16, relu=False)
        self.decode = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ConvBlock(16, 1, relu=False),
        )

    def fusion(self, feat1, feat2):
        concat_feat = concat_fusion(feat1, feat2)
        fused_feat = self.fuse(concat_feat)

        return fused_feat


class DBNet(nn.Module):
    '''A Dual-Branch Network for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(DBNet, self).__init__()
        self.conv = ConvBlock(1, 32)
        self.detail = nn.Sequential(
            ConvBlock(32, 16),
            DenseBlock(3, 16, 16),
        )
        self.semantic = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 64, stride=2),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )
        self.decode = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder(self, img):
        feat = self.conv(img)
        detail_feat = self.detail(feat)
        semantic_feat = self.semantic(feat)
        concat_feat = concat_fusion(detail_feat, semantic_feat)

        return concat_feat

    def fusion(self, feat1, feat2, mode='add'):
        if mode == 'add':
            return element_fusion(feat1, feat2, mode='add')
        elif mode == 'avg':
            return channel_fusion(feat1, feat2, mode='avg', softmax=False)
        else:
            raise ValueError("only supported ['add', 'avg'] mode")

    def decoder(self, feat):
        return self.decode(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feat = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img


class PFNetv1(nn.Module):
    '''PFNet: An Unsupervised Deep Network for Polarization Image Fusion'''
    def __init__(self):
        super(PFNetv1, self).__init__()
        self.encode1 = nn.Sequential(
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.encode2 = nn.Sequential(
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.decode = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder1(self, img):
        return self.encode1(img)

    def encoder2(self, img):
        return self.encode2(img)

    def fusion(self, feat1, feat2):
        return concat_fusion(feat1, feat2)

    def decoder(self, feat):
        return self.decode(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder1(img1)
        feat2 = self.encoder2(img2)

        # fuse
        fused_feat = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img


class PFNetv2(_FusionModel):
    '''Polarization Image Fusion with Self-Learned Fusion Strategy'''
    def __init__(self):
        super(PFNetv2, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.fuse = nn.Sequential(
            ConvBlock(2, 2),
            ConvBlock(2, 2),
            ConvBlock(2, 1, relu=False),
        )
        self.decode = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def fusion(self, feat1, feat2):
        fused_feat = []

        for i in range(feat1.shape[1]):
            concat_feat = torch.stack((feat1[:, i], feat2[:, i]), dim=1)
            fused_feat.append(self.fuse(concat_feat))

        return torch.cat(fused_feat, dim=1) + feat1 + feat2


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    models = (DeepFuse, DenseFuse, VIFNet, DIFNet, DBNet, PFNetv1, PFNetv2)

    model = models[5]()
    print(model)
    # summary(model, [(1, 224, 224), (1, 224, 224)], 2, device='cpu')

    x1 = torch.rand(2, 1, 224, 224)
    x2 = torch.rand(2, 1, 224, 224)

    outs = model(x1, x2)
    print(outs.shape)
