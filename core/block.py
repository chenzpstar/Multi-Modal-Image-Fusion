# -*- coding: utf-8 -*-
"""
# @file name  : block.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模块类
"""

import torch
import torch.nn as nn

try:
    from .fusion import concat_fusion
except:
    from fusion import concat_fusion

__all__ = [
    'ConvLayer', 'DeconvLayer', 'ResBlock', 'DenseBlock', 'SepConvBlock',
    'Res2Block', 'ConvFormerBlock', 'Res2FormerBlock', 'TransitionBlock',
    'ConvBlock', 'ECB', 'DCB', 'RFN', 'NestEncoder', 'NestDecoder', 'FSDecoder'
]


class ConvLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 ksize=3,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias=None,
                 norm=None,
                 pre_norm=None,
                 act='relu',
                 padding_mode='reflect'):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = ksize // 2
        if bias is None:
            bias = (norm != 'bn') or (pre_norm != 'bn')

        layers = []

        if pre_norm == 'bn':
            layers.append(nn.BatchNorm2d(in_ch))
        elif pre_norm == 'in':
            layers.append(nn.InstanceNorm2d(in_ch))
        elif pre_norm == 'ln':
            layers.append(LayerNorm(in_ch))

        layers.append(
            nn.Conv2d(in_ch,
                      out_ch,
                      ksize,
                      stride,
                      padding,
                      dilation,
                      groups,
                      bias=bias,
                      padding_mode=padding_mode))

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm == 'in':
            layers.append(nn.InstanceNorm2d(out_ch))
        elif norm == 'ln':
            layers.append(LayerNorm(out_ch))

        if act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == 'tanh':
            layers.append(nn.Tanh())

        self.conv = nn.Sequential(*layers)
        self.norm = norm
        self.pre_norm = pre_norm
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

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class DeconvLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 ksize=3,
                 stride=1,
                 padding=None,
                 output_padding=0,
                 bias=None,
                 norm=None,
                 act='relu',
                 padding_mode='zeros'):
        super(DeconvLayer, self).__init__()
        if padding is None:
            padding = ksize // 2
        if bias is None:
            bias = (norm != 'bn')

        layers = [
            nn.ConvTranspose2d(in_ch,
                               out_ch,
                               ksize,
                               stride,
                               padding,
                               output_padding,
                               bias=bias,
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

            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm1=None, norm2=None):
        super(ResBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            ConvLayer(in_ch, out_ch, norm=norm1),
            ConvLayer(out_ch, out_ch, norm=norm2, act=None),
        )

    def forward(self, x):
        return self.layers(x) + x


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=3):
        super(DenseBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.ModuleList(
            [ConvLayer(in_ch + i * out_ch, out_ch) for i in range(num_convs)])

    def forward(self, x):
        for layer in self.layers:
            x = concat_fusion((x, layer(x)))

        return x


class SepConvBlock(nn.Module):
    def __init__(self,
                 num_ch,
                 expand_ratio=2,
                 ksize=7,
                 bias=False,
                 act='relu',
                 residual=True,
                 attention=False):
        super(SepConvBlock, self).__init__()
        self.residual = residual
        self.attention = attention

        self.num_ch = num_ch
        self.expand_ratio = expand_ratio
        hid_ch = num_ch * expand_ratio

        self.pwconv1 = ConvLayer(num_ch, hid_ch, ksize=1, bias=bias, act=act)
        self.dwconv = ConvLayer(hid_ch,
                                hid_ch,
                                ksize=ksize,
                                groups=hid_ch,
                                bias=bias,
                                act=act)
        self.pwconv2 = ConvLayer(hid_ch, num_ch, ksize=1, bias=bias, act=None)

    def forward(self, x):
        if self.residual:
            res = x.clone()

        x = self.pwconv1(x)

        if self.attention:
            att = x.clone()

        x = self.dwconv(x)

        if self.attention:
            x *= att

        x = self.pwconv2(x)

        if self.residual:
            x += res

        return x


class Res2Block(nn.Module):
    def __init__(self,
                 num_ch,
                 width,
                 scale=4,
                 bias=False,
                 act='relu',
                 residual=True,
                 attention=False):
        super(Res2Block, self).__init__()
        self.residual = residual
        self.attention = attention

        self.num_ch = num_ch
        self.width = width
        self.scale = scale
        hid_ch = width * scale

        self.pwconv1 = ConvLayer(num_ch, hid_ch, ksize=1, bias=bias, act=act)
        self.num_convs = scale - 1 if scale > 1 else 1
        self.convs = nn.ModuleList([
            ConvLayer(width, width, bias=bias, act=act)
            for _ in range(self.num_convs)
        ])
        self.pwconv2 = ConvLayer(hid_ch, num_ch, ksize=1, bias=bias, act=None)

    def forward(self, x):
        if self.residual:
            res = x.clone()

        x = self.pwconv1(x)

        if self.attention:
            att = x.clone()

        xs = torch.chunk(x, self.scale, dim=1)

        if self.scale > 1:
            y = 0
            for i in range(self.num_convs):
                y += xs[i]
                y = self.convs[i](y)
                out = concat_fusion((out, y)) if i > 0 else y
            out = concat_fusion((out, xs[-1]))
        else:
            out = self.convs[0](xs[0])

        if self.attention:
            out *= att

        out = self.pwconv2(out)

        if self.residual:
            out += res

        return out


class MLP(nn.Module):
    def __init__(self, num_ch, mlp_ratio=4, bias=False, act='relu', drop=0.0):
        super(MLP, self).__init__()
        self.num_ch = num_ch
        self.mlp_ratio = mlp_ratio
        hid_ch = num_ch * mlp_ratio

        self.fc1 = ConvLayer(num_ch, hid_ch, ksize=1, bias=bias, act=act)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = ConvLayer(hid_ch, num_ch, ksize=1, bias=bias, act=None)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x))

        return x


class Scale(nn.Module):
    def __init__(self, num_ch, init_value=1.0, trainable=True):
        super(Scale, self).__init__()
        self.num_ch = num_ch
        self.init_value = init_value
        self.scale = nn.Parameter(init_value * torch.ones(num_ch),
                                  requires_grad=trainable)

    def forward(self, x):
        return self.scale.unsqueeze(-1).unsqueeze(-1) * x


class LayerNorm(nn.Module):
    def __init__(self,
                 affine_shape=None,
                 normalized_dim=(1, ),
                 scale=True,
                 bias=False,
                 eps=1e-6):
        super(LayerNorm, self).__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(
            (affine_shape, 1, 1))) if scale else None
        self.bias = nn.Parameter(torch.zeros(
            (affine_shape, 1, 1))) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)

        if self.use_scale:
            x *= self.weight

        if self.use_bias:
            x += self.bias

        return x


class MetaFormerBlock(nn.Module):
    def __init__(self,
                 num_ch,
                 token_mixer=nn.Identity,
                 norm_layer=LayerNorm,
                 layer_scale=None,
                 res_scale=None,
                 drop=0.0):
        super(MetaFormerBlock, self).__init__()
        self.num_ch = num_ch

        self.norm1 = norm_layer(num_ch)
        self.token_mixer = token_mixer(num_ch)
        self.layer_scale1 = Scale(
            num_ch, layer_scale) if layer_scale else nn.Identity()
        self.res_scale1 = Scale(num_ch,
                                res_scale) if res_scale else nn.Identity()

        self.norm2 = norm_layer(num_ch)
        self.mlp = MLP(num_ch, drop=drop)
        self.layer_scale2 = Scale(
            num_ch, layer_scale) if layer_scale else nn.Identity()
        self.res_scale2 = Scale(num_ch,
                                res_scale) if res_scale else nn.Identity()

    def forward(self, x):
        x = self.layer_scale1(self.token_mixer(
            self.norm1(x))) + self.res_scale1(x)
        x = self.layer_scale2(self.mlp(self.norm2(x))) + self.res_scale2(x)

        return x


class ConvFormerBlock(MetaFormerBlock):
    def __init__(self,
                 num_ch,
                 norm_layer=nn.BatchNorm2d,
                 layer_scale=None,
                 res_scale=None):
        super(ConvFormerBlock, self).__init__(num_ch,
                                              norm_layer=norm_layer,
                                              layer_scale=layer_scale,
                                              res_scale=res_scale)
        self.token_mixer = SepConvBlock(num_ch,
                                        residual=False,
                                        attention=False)


class Res2FormerBlock(MetaFormerBlock):
    def __init__(self,
                 num_ch,
                 width,
                 norm_layer=nn.BatchNorm2d,
                 layer_scale=None,
                 res_scale=None):
        super(Res2FormerBlock, self).__init__(num_ch,
                                              norm_layer=norm_layer,
                                              layer_scale=layer_scale,
                                              res_scale=res_scale)
        self.token_mixer = Res2Block(num_ch,
                                     width,
                                     residual=False,
                                     attention=False)


class TransitionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, down_mode='stride'):
        super(TransitionBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if down_mode == 'stride':
            self.layers = nn.Sequential(
                ConvLayer(in_ch, out_ch, ksize=1),
                ConvLayer(out_ch,
                          out_ch,
                          stride=stride,
                          groups=out_ch,
                          norm='bn'),
            )

        elif down_mode == 'maxpool':
            self.layers = nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvLayer(in_ch, out_ch, ksize=1, norm='bn'),
            )

    def forward(self, x):
        return self.layers(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize1=3, ksize2=1):
        super(ConvBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.layers = nn.Sequential(
            ConvLayer(in_ch, in_ch // 2, ksize=ksize1),
            ConvLayer(in_ch // 2, out_ch, ksize=ksize2),
        )

    def forward(self, x):
        return self.layers(x)


class ECB(ConvBlock):
    def __init__(self, in_ch, out_ch, ksize1=1, ksize2=3):
        super(ECB, self).__init__(in_ch, out_ch, ksize1=ksize1, ksize2=ksize2)


class DCB(ConvBlock):
    def __init__(self, in_ch, out_ch, ksize1=3, ksize2=3):
        super(DCB, self).__init__(in_ch, out_ch, ksize1=ksize1, ksize2=ksize2)


class RFN(nn.Module):
    def __init__(self, num_ch):
        super(RFN, self).__init__()
        self.res = ConvLayer(num_ch * 2, num_ch)

        self.conv1 = ConvLayer(num_ch, num_ch)
        self.conv2 = ConvLayer(num_ch, num_ch)

        self.layers = nn.Sequential(
            ConvLayer(num_ch * 2, num_ch, ksize=1),
            ConvLayer(num_ch, num_ch),
            ConvLayer(num_ch, num_ch),
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

        if down_mode == 'stride':
            self.down1 = ConvLayer(out_ch[1], out_ch[1], stride=2)
            self.down2 = ConvLayer(in_ch[2] * 2, in_ch[2] * 2, stride=2)
            self.down3 = ConvLayer(out_ch[2], out_ch[2], stride=2)

        elif down_mode == 'maxpool':
            self.down1 = nn.MaxPool2d(2, 2)
            self.down2 = nn.MaxPool2d(2, 2)
            self.down3 = nn.MaxPool2d(2, 2)

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
    def __init__(self, block, num_ch, up_mode='bilinear'):
        super(NestDecoder, self).__init__()
        self.DCB1_1 = block(num_ch[0] + num_ch[1], num_ch[0])
        self.DCB2_1 = block(num_ch[1] + num_ch[2], num_ch[1])
        self.DCB3_1 = block(num_ch[2] + num_ch[3], num_ch[2])

        self.DCB1_2 = block(num_ch[0] * 2 + num_ch[1], num_ch[0])
        self.DCB2_2 = block(num_ch[1] * 2 + num_ch[2], num_ch[1])

        self.DCB1_3 = block(num_ch[0] * 3 + num_ch[1], num_ch[0])

        self.up = Upsample(up_mode, 2)

    def forward(self, feats):
        x1_1 = self.DCB1_1(
            concat_fusion((feats[0], self.up(feats[1], feats[0].shape))))
        x2_1 = self.DCB2_1(
            concat_fusion((feats[1], self.up(feats[2], feats[1].shape))))
        x3_1 = self.DCB3_1(
            concat_fusion((feats[2], self.up(feats[3], feats[2].shape))))

        x1_2 = self.DCB1_2(
            concat_fusion((feats[0], x1_1, self.up(x2_1, x1_1.shape))))
        x2_2 = self.DCB2_2(
            concat_fusion((feats[1], x2_1, self.up(x3_1, x2_1.shape))))

        x1_3 = self.DCB1_3(
            concat_fusion((feats[0], x1_1, x1_2, self.up(x2_2, x1_2.shape))))

        return x1_3


class FSDecoder(nn.Module):
    def __init__(self, block, num_ch, up_mode='bilinear'):
        super(FSDecoder, self).__init__()
        self.DB1 = block(128, num_ch[0])
        self.DB2 = block(128, num_ch[1])
        self.DB3 = block(128, num_ch[2])

        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(4, 4)

        self.down1_2 = ConvLayer(num_ch[0], 32, ksize=1)
        self.down1_3 = ConvLayer(num_ch[0], 32, ksize=1)
        self.down2_3 = ConvLayer(num_ch[1], 32, ksize=1)

        self.lateral1 = ConvLayer(num_ch[0], 32, ksize=1)
        self.lateral2 = ConvLayer(num_ch[1], 32, ksize=1)
        self.lateral3 = ConvLayer(num_ch[2], 32, ksize=1)

        self.up1 = Upsample(up_mode, 2)
        self.up2 = Upsample(up_mode, 4)
        self.up3 = Upsample(up_mode, 8)

        self.up4_3 = ConvLayer(num_ch[3], 32, ksize=1)
        self.up4_2 = ConvLayer(num_ch[3], 32, ksize=1)
        self.up4_1 = ConvLayer(num_ch[3], 32, ksize=1)
        self.up3_2 = ConvLayer(num_ch[2], 32, ksize=1)
        self.up3_1 = ConvLayer(num_ch[2], 32, ksize=1)
        self.up2_1 = ConvLayer(num_ch[1], 32, ksize=1)

    def forward(self, feats):
        x1_1 = self.down1_3(self.down2(feats[0]))
        x2_1 = self.down2_3(self.down1(feats[1]))
        x3_1 = self.lateral3(feats[2])
        x4_1 = self.up4_3(self.up1(feats[3], feats[2].shape))
        y3 = self.DB3(concat_fusion((x1_1, x2_1, x3_1, x4_1)))

        x1_2 = self.down1_2(self.down1(feats[0]))
        x2_2 = self.lateral2(feats[1])
        x3_2 = self.up3_2(self.up1(y3, feats[1].shape))
        x4_2 = self.up4_2(self.up2(feats[3], feats[1].shape))
        y2 = self.DB2(concat_fusion((x1_2, x2_2, x3_2, x4_2)))

        x1_3 = self.lateral1(feats[0])
        x2_3 = self.up2_1(self.up1(y2, feats[0].shape))
        x3_3 = self.up3_1(self.up2(y3, feats[0].shape))
        x4_3 = self.up4_1(self.up3(feats[3], feats[0].shape))
        y1 = self.DB1(concat_fusion((x1_3, x2_3, x3_3, x4_3)))

        return y1


class Upsample(nn.Module):
    def __init__(self, mode, scale_factor=2):
        super(Upsample, self).__init__()
        if mode == 'nearest':
            self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=scale_factor,
                                  mode=mode,
                                  align_corners=True)

    def forward(self, feat, shape):
        out = self.up(feat)

        if out.shape != shape:
            out = self._pad(out, shape)

        return out

    @staticmethod
    def _pad(feat, shape):
        pad_h = shape[-2] - feat.shape[-2]
        pad_w = shape[-1] - feat.shape[-1]
        pad_h1, pad_w1 = pad_h // 2, pad_w // 2
        pad_h2, pad_w2 = pad_h - pad_h1, pad_w - pad_w1
        padding = (pad_w1, pad_w2, pad_h1, pad_h2)

        return nn.ReflectionPad2d(padding)(feat)
