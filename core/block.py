# -*- coding: utf-8 -*-
"""
# @file name  : block.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模块类
"""

import torch.nn as nn

try:
    from .fusion import concat_fusion
except:
    from fusion import concat_fusion

__all__ = [
    'ConvBlock', 'DeconvBlock', 'ResBlock', 'DenseBlock', 'CB', 'ECB', 'DCB',
    'RFN', 'NestEncoder', 'NestDecoder'
]


class ConvBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 ksize=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 norm=None,
                 act='relu',
                 padding_mode='zeros'):
        super(ConvBlock, self).__init__()
        padding = 1 if ksize == 3 else ksize // 2
        layers = [
            nn.Conv2d(in_ch,
                      out_ch,
                      ksize,
                      stride,
                      padding,
                      dilation,
                      groups,
                      bias=(norm != 'bn'),
                      padding_mode=padding_mode)
        ]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'in':
            layers.append(nn.InstanceNorm2d(out_ch))

        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == 'tanh':
            layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)
        self.norm = norm
        self.act = act
        self._init_weights()

    def forward(self, x):
        return self.conv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.act == 'relu':
                    nn.init.kaiming_normal_(m.weight)
                elif self.act == 'lrelu':
                    nn.init.kaiming_normal_(m.weight, a=0.2)
                elif self.act == 'tanh':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DeconvBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 ksize=3,
                 stride=1,
                 output_padding=0,
                 norm=None,
                 act='relu',
                 padding_mode='zeros'):
        super(DeconvBlock, self).__init__()
        padding = 1 if ksize == 3 else ksize // 2
        layers = [
            nn.ConvTranspose2d(in_ch,
                               out_ch,
                               ksize,
                               stride,
                               padding,
                               output_padding,
                               bias=(norm != 'bn'),
                               padding_mode=padding_mode)
        ]

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'in':
            layers.append(nn.InstanceNorm2d(out_ch))

        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == 'tanh':
            layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)
        self.norm = norm
        self.act = act
        self._init_weights()

    def forward(self, x):
        return self.deconv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if self.act == 'relu':
                    nn.init.kaiming_normal_(m.weight)
                elif self.act == 'lrelu':
                    nn.init.kaiming_normal_(m.weight, a=0.2)
                elif self.act == 'tanh':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm1=None, norm2=None):
        super(ResBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            ConvBlock(in_ch, out_ch, norm=norm1),
            ConvBlock(out_ch, out_ch, norm=norm2, act=None),
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
            x = concat_fusion((layer(x), x))

        return x


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_ConvBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential()

    def forward(self, x):
        return self.layers(x)


class CB(_ConvBlock):
    def __init__(self, in_ch, out_ch):
        super(CB, self).__init__(in_ch, out_ch)
        self.layers = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2),
            ConvBlock(in_ch // 2, out_ch, ksize=1),
        )


class ECB(_ConvBlock):
    def __init__(self, in_ch, out_ch):
        super(ECB, self).__init__(in_ch, out_ch)
        self.layers = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2, ksize=1),
            ConvBlock(in_ch // 2, out_ch),
        )


class DCB(_ConvBlock):
    def __init__(self, in_ch, out_ch):
        super(DCB, self).__init__(in_ch, out_ch)
        self.layers = nn.Sequential(
            ConvBlock(in_ch, in_ch // 2),
            ConvBlock(in_ch // 2, out_ch),
        )


class RFN(nn.Module):
    def __init__(self, num_ch):
        super(RFN, self).__init__()
        self.num_ch = num_ch

        self.res = ConvBlock(num_ch * 2, num_ch)

        self.conv1 = ConvBlock(num_ch, num_ch)
        self.conv2 = ConvBlock(num_ch, num_ch)

        self.layers = nn.Sequential(
            ConvBlock(num_ch * 2, num_ch, ksize=1),
            ConvBlock(num_ch, num_ch),
            ConvBlock(num_ch, num_ch),
        )

    def forward(self, x1, x2):
        f_res = self.res(concat_fusion((x1, x2)))

        f1 = self.conv1(x1)
        f2 = self.conv2(x2)
        f_out = self.layers(concat_fusion((f1, f2)))

        return f_out + f_res


class NestEncoder(nn.Module):
    def __init__(self, block, in_ch, out_ch, down_mode='stride'):
        super(NestEncoder, self).__init__()
        self.ECB2_1 = block(in_ch[1] + in_ch[0], out_ch[1])
        self.ECB3_1 = block(in_ch[2] + in_ch[1], in_ch[2] * 2)
        self.ECB4_1 = block(in_ch[3] + in_ch[2], in_ch[3] * 2)

        self.ECB3_2 = block(in_ch[2] * 3 + out_ch[1], out_ch[2])
        self.ECB4_2 = block(in_ch[3] * 3 + in_ch[2] * 2,
                            in_ch[3] * 4 + in_ch[2])

        self.ECB4_3 = block(in_ch[3] * 7 + in_ch[2] + out_ch[2], out_ch[3])

        self.down1 = ConvBlock(
            out_ch[1], out_ch[1],
            stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)
        self.down2 = ConvBlock(
            in_ch[2] * 2, in_ch[2] *
            2, stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)
        self.down3 = ConvBlock(
            out_ch[2], out_ch[2],
            stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)

    def forward(self, feats):
        x2_1 = self.ECB2_1(concat_fusion(feats[1]))
        x3_1 = self.ECB3_1(concat_fusion(feats[2]))
        x4_1 = self.ECB4_1(concat_fusion(feats[3]))

        x3_2 = self.ECB3_2(concat_fusion(
            (feats[2][0], x3_1, self.down1(x2_1))))
        x4_2 = self.ECB4_2(concat_fusion(
            (feats[3][0], x4_1, self.down2(x3_1))))

        x4_3 = self.ECB4_3(
            concat_fusion((feats[3][0], x4_1, x4_2, self.down3(x3_2))))

        return feats[0], x2_1, x3_2, x4_3


class NestDecoder(nn.Module):
    def __init__(self, block, num_ch, up_mode='nearest'):
        super(NestDecoder, self).__init__()
        self.DCB1_1 = block(num_ch[0] + num_ch[1], num_ch[0])
        self.DCB2_1 = block(num_ch[1] + num_ch[2], num_ch[1])
        self.DCB3_1 = block(num_ch[2] + num_ch[3], num_ch[2])

        self.DCB1_2 = block(num_ch[0] * 2 + num_ch[1], num_ch[0])
        self.DCB2_2 = block(num_ch[1] * 2 + num_ch[2], num_ch[1])

        self.DCB1_3 = block(num_ch[0] * 3 + num_ch[1], num_ch[0])

        if up_mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=up_mode)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                  mode=up_mode,
                                  align_corners=True)

    def forward(self, feats):
        x1_1 = self.DCB1_1(concat_fusion((feats[0], self.up(feats[1]))))
        x2_1 = self.DCB2_1(concat_fusion((feats[1], self.up(feats[2]))))
        x3_1 = self.DCB3_1(concat_fusion((feats[2], self.up(feats[3]))))

        x1_2 = self.DCB1_2(concat_fusion((feats[0], x1_1, self.up(x2_1))))
        x2_2 = self.DCB2_2(concat_fusion((feats[1], x2_1, self.up(x3_1))))

        x1_3 = self.DCB1_3(concat_fusion(
            (feats[0], x1_1, x1_2, self.up(x2_2))))

        return x1_3
