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
    from .fusion import *
except:
    from fusion import *


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bn=False,
                 relu=True):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch,
                      out_ch,
                      kernel_size,
                      stride,
                      padding,
                      bias=(bn is False))
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)
        self._init_weights()

    def forward(self, x):
        return self.conv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_ch, out_ch):
        super(DenseBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.ModuleList(
            [ConvBlock(in_ch + i * out_ch, out_ch) for i in range(num_convs)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat((x, layer(x)), dim=1)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            ConvBlock(in_ch, out_ch, bn=True),
            ConvBlock(out_ch, out_ch, relu=False),
        )

    def forward(self, x):
        return self.layers(x) + x


class DeepFuse(nn.Module):
    '''DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs'''
    def __init__(self):
        super(DeepFuse, self).__init__()
        self.convs1 = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5, padding=2),
            ConvBlock(16, 32, kernel_size=7, padding=3),
        )

        self.convs2 = nn.Sequential(
            ConvBlock(32, 32, kernel_size=7, padding=3),
            ConvBlock(32, 16, kernel_size=5, padding=2),
            ConvBlock(16, 1, kernel_size=5, padding=2, relu=False),
        )

    def encoder(self, img):
        return self.convs1(img)

    def fusion(self, feat1, feat2, mode='add'):
        return element_fusion(feat1, feat2, mode)

    def decoder(self, feat):
        return self.convs2(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


class DenseFuse(nn.Module):
    '''DenseFuse: A Fusion Approach to Infrared and Visible Images'''
    def __init__(self):
        super(DenseFuse, self).__init__()
        self.conv = ConvBlock(1, 16)
        self.dense = DenseBlock(3, 16, 16)

        self.convs = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder(self, img):
        feat = self.conv(img)
        feat = self.dense(feat)

        return feat

    def fusion(self, feat1, feat2, mode='add'):
        if mode == 'add':
            return element_fusion(feat1, feat2, mode='add')
        elif mode == 'l1':
            return spatial_fusion(feat1, feat2, mode='l1', softmax=False)
        else:
            raise ValueError("only supported ['add', 'l1'] mode")

    def decoder(self, feat):
        return self.convs(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


class VIFNet(nn.Module):
    '''VIF-Net: An Unsupervised Framework for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(VIFNet, self).__init__()
        self.conv = ConvBlock(1, 16)
        self.dense = DenseBlock(3, 16, 16)

        self.convs = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder(self, img):
        feat = self.conv(img)
        feat = self.dense(feat)

        return feat

    def fusion(self, feat1, feat2):
        return concat_fusion(feat1, feat2)

    def decoder(self, feat):
        return self.convs(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


class DIFNet(nn.Module):
    '''Unsupervised Deep Image Fusion With Structure Tensor Representations'''
    def __init__(self):
        super(DIFNet, self).__init__()
        self.conv = ConvBlock(1, 16)
        self.res1 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
        )

        self.fuse = ConvBlock(32, 16, relu=False)

        self.res2 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder(self, img):
        feat = self.conv(img)
        feat = self.res1(feat)

        return feat

    def fusion(self, feat1, feat2):
        concat_feats = concat_fusion(feat1, feat2)
        fused_feats = self.fuse(concat_feats)

        return fused_feats

    def decoder(self, feat):
        return self.res2(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


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

        self.convs = nn.Sequential(
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
        return self.convs(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


class PFNetv1(nn.Module):
    '''PFNet: An Unsupervised Deep Network for Polarization Image Fusion'''
    def __init__(self):
        super(PFNetv1, self).__init__()
        self.conv1 = ConvBlock(1, 16)
        self.conv2 = ConvBlock(1, 16)

        self.dense1 = DenseBlock(3, 16, 16)
        self.dense2 = DenseBlock(3, 16, 16)

        self.convs = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder1(self, img):
        feat = self.conv1(img)
        feat = self.dense1(feat)

        return feat

    def encoder2(self, img):
        feat = self.conv2(img)
        feat = self.dense2(feat)

        return feat

    def fusion(self, feat1, feat2):
        return concat_fusion(feat1, feat2)

    def decoder(self, feat):
        return self.convs(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder1(img1)
        feat2 = self.encoder2(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


class PFNetv2(nn.Module):
    '''Polarization Image Fusion with Self-Learned Fusion Strategy'''
    def __init__(self):
        super(PFNetv2, self).__init__()
        self.conv = ConvBlock(1, 16)
        self.dense = DenseBlock(3, 16, 16)

        self.fuse = nn.Sequential(
            ConvBlock(2, 2),
            ConvBlock(2, 2),
            ConvBlock(2, 1, relu=False),
        )

        self.convs = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def encoder(self, img):
        feat = self.conv(img)
        feat = self.dense(feat)

        return feat

    def fusion(self, feat1, feat2):
        fused_feat = []

        for i in range(feat1.shape[1]):
            concat_feat = torch.stack((feat1[:, i], feat2[:, i]), dim=1)
            fused_feat.append(self.fuse(concat_feat))

        return torch.cat(fused_feat, dim=1) + feat1 + feat2

    def decoder(self, feat):
        return self.convs(feat)

    def forward(self, img1, img2):
        # extract
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)

        # fuse
        fused_feats = self.fusion(feat1, feat2)

        # reconstruct
        fused_img = self.decoder(fused_feats)

        return fused_img


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
