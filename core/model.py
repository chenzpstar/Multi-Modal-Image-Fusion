# -*- coding: utf-8 -*-
"""
# @file name  : model.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模型类
"""

import torch
import torch.nn as nn


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

        layers = []

        for _ in range(num_convs):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch += out_ch

        self.layers = nn.ModuleList(layers)

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

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.convs1(vis)
        conv_dolp = self.convs1(dolp)

        # fuse
        fused_feats = conv_vis + conv_dolp

        # reconstruct
        fused_img = self.convs2(fused_feats)

        return fused_img


class DenseFuse(nn.Module):
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

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.conv(vis)
        conv_dolp = self.conv(dolp)

        dense_vis = self.dense(conv_vis)
        dense_dolp = self.dense(conv_dolp)

        # fuse
        fused_feats = dense_vis + dense_dolp

        # reconstruct
        fused_img = self.convs(fused_feats)

        return fused_img


class DIFNet(nn.Module):
    def __init__(self):
        super(DIFNet, self).__init__()
        self.conv1 = ConvBlock(1, 16)
        self.res_convs1 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
        )

        self.conv2 = ConvBlock(32, 16, relu=False)
        self.res_convs2 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ConvBlock(16, 1, relu=False),
        )

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.conv1(vis)
        conv_dolp = self.conv1(dolp)

        res_vis = self.res_convs1(conv_vis)
        res_dolp = self.res_convs1(conv_dolp)

        # fuse
        concat_feats = torch.cat((res_vis, res_dolp), dim=1)
        fused_feats = self.conv2(concat_feats)

        # reconstruct
        fused_img = self.res_convs2(fused_feats)

        return fused_img


class VIFNet(nn.Module):
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

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.conv(vis)
        conv_dolp = self.conv(dolp)

        dense_vis = self.dense(conv_vis)
        dense_dolp = self.dense(conv_dolp)

        # fuse
        fused_feats = torch.cat((dense_vis, dense_dolp), dim=1)

        # reconstruct
        fused_img = self.convs(fused_feats)

        return fused_img


class PFNetv1(nn.Module):
    def __init__(self):
        super(PFNetv1, self).__init__()
        self.conv_vis = ConvBlock(1, 16)
        self.conv_dolp = ConvBlock(1, 16)

        self.dense_vis = DenseBlock(3, 16, 16)
        self.dense_dolp = DenseBlock(3, 16, 16)

        self.convs = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, relu=False),
        )

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.conv_vis(vis)
        conv_dolp = self.conv_dolp(dolp)

        dense_vis = self.dense_vis(conv_vis)
        dense_dolp = self.dense_dolp(conv_dolp)

        # fuse
        fused_feats = torch.cat((dense_vis, dense_dolp), dim=1)

        # reconstruct
        fused_img = self.convs(fused_feats)

        return fused_img


class PFNetv2(nn.Module):
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

    def forward(self, vis, dolp):
        # extract
        conv_vis = self.conv(vis)
        conv_dolp = self.conv(dolp)

        dense_vis = self.dense(conv_vis)
        dense_dolp = self.dense(conv_dolp)

        # fuse
        fused_feats = []

        for i in range(dense_vis.shape[1]):
            concat_feat = torch.stack((dense_vis[:, i], dense_dolp[:, i]),
                                      dim=1)
            fused_feats.append(self.fuse(concat_feat))

        fused_feats = torch.cat(fused_feats, dim=1) + dense_vis + dense_dolp

        # reconstruct
        fused_img = self.convs(fused_feats)

        return fused_img


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    models = (DeepFuse, DenseFuse, DIFNet, VIFNet, PFNetv1, PFNetv2)

    model = models[4]()
    print(model)
    # summary(model, [(1, 224, 224), (1, 224, 224)], 2, device='cpu')

    x1 = torch.rand(2, 1, 224, 224)
    x2 = torch.rand(2, 1, 224, 224)

    outs = model(x1, x2)
    print(outs.shape)
