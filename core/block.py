# -*- coding: utf-8 -*-
"""
# @file name  : model.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模块类
"""

import torch
import torch.nn as nn

__all__ = ['ConvBlock', 'ResBlock', 'DenseBlock']


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
