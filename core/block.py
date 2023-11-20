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
    'ConvLayer', 'ResBlock', 'DenseBlock', 'SepConvBlock', 'MixConvBlock',
    'Res2ConvBlock', 'Attention', 'ConvFormerBlock', 'MixFormerBlock',
    'Res2FormerBlock', 'TransformerBlock', 'TransitionBlock', 'DCBlock',
    'ConvBlock', 'ECB', 'DCB', 'RFN', 'NestEncoder', 'LSDecoder',
    'NestDecoder', 'FSDecoder', 'Downsample', 'Upsample'
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
                 layer=nn.Conv2d,
                 act=nn.ReLU,
                 padding_mode='reflect'):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = ksize // 2
        if bias is None:
            bias = (norm is not nn.BatchNorm2d) or (pre_norm
                                                    is not nn.BatchNorm2d)

        layers = []

        if pre_norm is not None:
            if pre_norm is nn.GroupNorm:
                layers.append(pre_norm(out_ch, out_ch))
            else:
                layers.append(pre_norm(out_ch))

        if layer is nn.Conv2d:
            layers.append(
                layer(in_ch,
                      out_ch,
                      ksize,
                      stride,
                      padding,
                      dilation,
                      groups,
                      bias=bias,
                      padding_mode=padding_mode))
        elif layer is nn.ConvTranspose2d:
            layers.append(
                layer(in_ch,
                      out_ch,
                      ksize,
                      stride,
                      padding,
                      output_padding=0,
                      bias=bias,
                      padding_mode='zeros'))

        if norm is not None:
            if norm is nn.GroupNorm:
                layers.append(norm(out_ch, out_ch))
            else:
                layers.append(norm(out_ch))

        if act is not None:
            if act in (nn.ReLU, nn.ReLU6, nn.Hardswish):
                layers.append(act(inplace=True))
            elif act is nn.LeakyReLU:
                layers.append(act(0.2, inplace=True))
            else:
                layers.append(act())

        self.layers = nn.Sequential(*layers)
        self.norm = norm
        self.pre_norm = pre_norm
        self.act = act
        self._init_weights()

    def forward(self, x):
        return self.layers(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.act in (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU):
                    nn.init.kaiming_normal_(m.weight)
                elif self.act is nn.LeakyReLU:
                    nn.init.kaiming_normal_(m.weight, a=0.2)
                elif self.act is nn.Tanh:
                    nn.init.xavier_normal_(m.weight,
                                           gain=nn.init.calculate_gain('tanh'))

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    '''Used in SEDRFuse, DIFNet'''
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
    '''Used in PFNet, DenseFuse, VIFNet, DBNet'''
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
                 in_ch,
                 out_ch,
                 scale=4,
                 ksize=3,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 residual=True,
                 attention=False):
        super(SepConvBlock, self).__init__()
        self.norm = norm
        self.act = act()
        self.residual = residual
        self.attention = attention

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale = scale
        hid_ch = in_ch * scale

        self.pwconv1 = ConvLayer(in_ch,
                                 hid_ch,
                                 ksize=1,
                                 bias=bias,
                                 norm=norm,
                                 act=act)
        self.dwconv = ConvLayer(hid_ch,
                                hid_ch,
                                ksize=ksize,
                                groups=hid_ch,
                                bias=bias,
                                norm=norm,
                                act=None)
        self.pwconv2 = ConvLayer(hid_ch,
                                 out_ch,
                                 ksize=1,
                                 bias=bias,
                                 norm=norm,
                                 act=None)

        if attention:
            self.pwconv = ConvLayer(in_ch,
                                    hid_ch,
                                    ksize=1,
                                    bias=bias,
                                    norm=norm,
                                    act=act)

        if residual:
            self.shortcut = ConvLayer(
                in_ch, out_ch, ksize=1, bias=bias, norm=norm,
                act=None) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        if self.residual:
            res = self.shortcut(x.clone())

        if self.attention:
            attn = self.pwconv(x.clone())

        out = self.dwconv(self.pwconv1(x))

        if self.attention:
            out *= attn

        out = self.pwconv2(out)

        if self.residual:
            out += res

        return self.act(out)


class MixConvBlock(SepConvBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 scale=4,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 residual=True,
                 attention=False):
        super(MixConvBlock, self).__init__(in_ch,
                                           out_ch,
                                           scale=scale,
                                           bias=bias,
                                           norm=norm,
                                           act=act,
                                           residual=residual,
                                           attention=attention)
        width = in_ch

        self.dwconvs = nn.ModuleList([
            ConvLayer(width,
                      width,
                      ksize=2 * i + 1,
                      groups=width,
                      bias=bias,
                      norm=norm,
                      act=None) for i in range(scale)
        ])

    def forward(self, x):
        if self.residual:
            res = self.shortcut(x.clone())

        if self.attention:
            attn = self.pwconv(x.clone())

        xs = torch.chunk(self.pwconv1(x), self.scale, dim=1)

        if self.scale > 1:
            for i in range(self.scale):
                y = self.dwconvs[i](xs[i])
                out = concat_fusion((out, y)) if i > 0 else y
        else:
            out = self.dwconvs[0](xs[0])

        if self.attention:
            out *= attn

        out = self.pwconv2(out)

        if self.residual:
            out += res

        return self.act(out)


class Res2ConvBlock(SepConvBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 scale=4,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 residual=True,
                 attention=False):
        super(Res2ConvBlock, self).__init__(in_ch,
                                            out_ch,
                                            scale=scale,
                                            bias=bias,
                                            norm=norm,
                                            act=act,
                                            residual=residual,
                                            attention=attention)
        width = in_ch

        # self.dwconvs = nn.ModuleList([
        #     ConvLayer(width,
        #               width,
        #               ksize=3,
        #               groups=width,
        #               bias=bias,
        #               norm=norm,
        #               act=None) if i > 0 else nn.Identity()
        #     for i in range(scale)
        # ])
        self.dwconvs = nn.ModuleList([
            ConvLayer(width,
                      width,
                      ksize=3 if i > 0 else 1,
                      groups=width,
                      bias=bias,
                      norm=norm,
                      act=None) for i in range(scale)
        ])

    def forward(self, x):
        if self.residual:
            res = self.shortcut(x.clone())

        if self.attention:
            attn = self.pwconv(x.clone())

        xs = torch.chunk(self.pwconv1(x), self.scale, dim=1)

        if self.scale > 1:
            for i in range(self.scale):
                y = y + xs[i] if i > 1 else xs[i]
                y = self.dwconvs[i](y)
                out = concat_fusion((out, y)) if i > 0 else y
        else:
            out = self.dwconvs[0](xs[0])

        if self.attention:
            out *= attn

        out = self.pwconv2(out)

        if self.residual:
            out += res

        return self.act(out)


class Attention(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 num_heads=None,
                 qkv_bias=False,
                 proj_bias=False,
                 norm=None,
                 act=None,
                 sr_ratio=None,
                 down_mode='stride'):
        super(Attention, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.num_heads = num_heads if num_heads else in_ch // 16
        self.head_dim = in_ch // self.num_heads
        self.att_dim = self.num_heads * self.head_dim

        self.scale = self.head_dim**-0.5

        self.q = ConvLayer(in_ch,
                           self.att_dim,
                           ksize=1,
                           bias=qkv_bias,
                           norm=norm,
                           act=act)
        self.k = ConvLayer(in_ch,
                           self.att_dim,
                           ksize=1,
                           bias=qkv_bias,
                           norm=norm,
                           act=act)
        self.v = ConvLayer(in_ch,
                           self.att_dim,
                           ksize=1,
                           bias=qkv_bias,
                           norm=norm,
                           act=act)
        self.proj = ConvLayer(self.att_dim,
                              out_ch,
                              ksize=1,
                              bias=proj_bias,
                              norm=norm,
                              act=act)

        self.sr_ratio = sr_ratio if sr_ratio else 16 // (in_ch // 16)

        if down_mode == 'stride':
            self.pool = ConvLayer(in_ch,
                                  in_ch,
                                  ksize=self.sr_ratio,
                                  stride=self.sr_ratio,
                                  padding=0,
                                  groups=in_ch,
                                  bias=False,
                                  norm=norm,
                                  act=act)
        elif down_mode == 'avgpool':
            self.pool = nn.AvgPool2d(self.sr_ratio, self.sr_ratio)

    def forward(self, x):
        b, _, h, w = x.shape

        q = self.q(x).flatten(2).reshape(b, self.num_heads, self.head_dim,
                                         -1).permute(0, 1, 3, 2)

        x_pool = self.pool(x) if self.sr_ratio > 1 else x
        k = self.k(x_pool).flatten(2).reshape(b, self.num_heads, self.head_dim,
                                              -1).permute(0, 1, 2, 3)
        v = self.v(x_pool).flatten(2).reshape(b, self.num_heads, self.head_dim,
                                              -1).permute(0, 1, 3, 2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(2, 3).reshape(b, self.att_dim, h, w)
        out = self.proj(out)

        return out


class FFN(nn.Module):
    def __init__(self,
                 num_ch,
                 scale=4,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6):
        super(FFN, self).__init__()
        self.num_ch = num_ch
        self.scale = scale
        hid_ch = num_ch * scale

        self.layers = nn.Sequential(
            ConvLayer(num_ch, hid_ch, ksize=1, bias=bias, norm=norm, act=act),
            ConvLayer(hid_ch,
                      hid_ch,
                      ksize=3,
                      groups=hid_ch,
                      bias=bias,
                      norm=norm,
                      act=act),
            ConvLayer(hid_ch, num_ch, ksize=1, bias=bias, norm=norm, act=None),
        )

    def forward(self, x):
        return self.layers(x)


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
                 in_ch,
                 out_ch,
                 token_mixer=nn.Identity,
                 norm_layer=LayerNorm,
                 act_layer=nn.Identity,
                 layer_scale=None,
                 res_scale=None):
        super(MetaFormerBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = norm_layer(in_ch)
        self.token_mixer = token_mixer(in_ch, out_ch)
        self.layer_scale1 = Scale(
            out_ch, layer_scale) if layer_scale else nn.Identity()
        self.res_scale1 = Scale(out_ch,
                                res_scale) if res_scale else nn.Identity()

        self.norm2 = norm_layer(out_ch)
        self.ffn = FFN(out_ch)
        self.layer_scale2 = Scale(
            out_ch, layer_scale) if layer_scale else nn.Identity()
        self.res_scale2 = Scale(out_ch,
                                res_scale) if res_scale else nn.Identity()

        self.act = act_layer()

    def forward(self, x):
        out = self.act(
            self.layer_scale1(self.token_mixer(self.norm1(x))) +
            self.res_scale1(x))
        out = self.act(
            self.layer_scale2(self.ffn(self.norm2(out))) +
            self.res_scale2(out))

        return out


class ConvFormerBlock(MetaFormerBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 layer_scale=None,
                 res_scale=None):
        super(ConvFormerBlock, self).__init__(in_ch,
                                              out_ch,
                                              norm_layer=norm_layer,
                                              act_layer=act_layer,
                                              layer_scale=layer_scale,
                                              res_scale=res_scale)
        self.token_mixer = SepConvBlock(in_ch,
                                        out_ch,
                                        residual=False,
                                        attention=False)


class MixFormerBlock(MetaFormerBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 layer_scale=None,
                 res_scale=None):
        super(MixFormerBlock, self).__init__(in_ch,
                                             out_ch,
                                             norm_layer=norm_layer,
                                             act_layer=act_layer,
                                             layer_scale=layer_scale,
                                             res_scale=res_scale)
        self.token_mixer = MixConvBlock(in_ch,
                                        out_ch,
                                        residual=False,
                                        attention=False)


class Res2FormerBlock(MetaFormerBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 layer_scale=None,
                 res_scale=None):
        super(Res2FormerBlock, self).__init__(in_ch,
                                              out_ch,
                                              norm_layer=norm_layer,
                                              act_layer=act_layer,
                                              layer_scale=layer_scale,
                                              res_scale=res_scale)
        self.token_mixer = Res2ConvBlock(in_ch,
                                         out_ch,
                                         residual=False,
                                         attention=False)


class TransformerBlock(MetaFormerBlock):
    def __init__(self,
                 in_ch,
                 out_ch,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU6,
                 layer_scale=None,
                 res_scale=None):
        super(TransformerBlock, self).__init__(in_ch,
                                               out_ch,
                                               norm_layer=norm_layer,
                                               act_layer=act_layer,
                                               layer_scale=layer_scale,
                                               res_scale=res_scale)
        self.token_mixer = Attention(in_ch, out_ch)


class TransitionBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 stride=2,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 down_mode='stride'):
        super(TransitionBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        if down_mode == 'stride':
            self.layers = nn.Sequential(
                ConvLayer(in_ch,
                          in_ch,
                          ksize=stride,
                          stride=stride,
                          padding=0,
                          groups=in_ch,
                          bias=bias,
                          norm=norm,
                          act=act),
                ConvLayer(in_ch,
                          out_ch,
                          ksize=1,
                          bias=bias,
                          norm=norm,
                          act=act),
            )

        elif down_mode == 'maxpool':
            self.layers = nn.Sequential(
                nn.MaxPool2d(stride, stride),
                ConvLayer(in_ch,
                          out_ch,
                          ksize=1,
                          bias=bias,
                          norm=norm,
                          act=act),
            )

    def forward(self, x):
        return self.layers(x)


class DCBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 residual=False):
        super(DCBlock, self).__init__()
        self.residual = residual

        self.in_ch = in_ch
        self.out_ch = out_ch
        hid_ch = in_ch // 2

        self.layers = nn.Sequential(
            ConvLayer(in_ch, hid_ch, ksize=1, bias=bias, norm=norm, act=act),
            ConvLayer(hid_ch,
                      hid_ch,
                      ksize=3,
                      groups=hid_ch,
                      bias=bias,
                      norm=norm,
                      act=act),
            ConvLayer(hid_ch, out_ch, ksize=1, bias=bias, norm=norm, act=None),
        )

        if residual:
            self.shortcut = ConvLayer(
                in_ch, out_ch, ksize=1, bias=bias, norm=norm,
                act=None) if in_ch != out_ch else nn.Identity()
        
        self.act = act()

    def forward(self, x):
        if self.residual:
            return self.act(self.layers(x) + self.shortcut(x))

        return self.act(self.layers(x))


class ConvBlock(nn.Module):
    '''Used in NestFuse, RFNNest, MAFusion'''
    def __init__(self, in_ch, out_ch, ksize1=3, ksize2=1):
        super(ConvBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        hid_ch = in_ch // 2

        self.layers = nn.Sequential(
            ConvLayer(in_ch, hid_ch, ksize=ksize1),
            ConvLayer(hid_ch, out_ch, ksize=ksize2),
        )

    def forward(self, x):
        return self.layers(x)


class ECB(ConvBlock):
    '''Used in UNFusion'''
    def __init__(self, in_ch, out_ch, ksize1=1, ksize2=3):
        super(ECB, self).__init__(in_ch, out_ch, ksize1=ksize1, ksize2=ksize2)


class DCB(ConvBlock):
    '''Used in UNFusion'''
    def __init__(self, in_ch, out_ch, ksize1=3, ksize2=3):
        super(DCB, self).__init__(in_ch, out_ch, ksize1=ksize1, ksize2=ksize2)


class RFN(nn.Module):
    '''Used in RFNNest'''
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
    '''Used in UNFusion'''
    def __init__(self, block, in_ch, out_ch, down_mode='stride'):
        super(NestEncoder, self).__init__()
        self.EB2_1 = block(in_ch[1] + in_ch[0], out_ch[1])
        self.EB3_1 = block(in_ch[2] + in_ch[1], in_ch[2] * 2)
        self.EB4_1 = block(in_ch[3] + in_ch[2], in_ch[3] * 2)

        self.EB3_2 = block(in_ch[2] * 3 + out_ch[1], out_ch[2])
        self.EB4_2 = block(in_ch[3] * 3 + in_ch[2] * 2,
                           in_ch[3] * 4 + in_ch[2])

        self.EB4_3 = block(in_ch[3] * 7 + in_ch[2] + out_ch[2], out_ch[3])

        if down_mode == 'stride':
            self.down1 = ConvLayer(out_ch[1], out_ch[1], stride=2)
            self.down2 = ConvLayer(in_ch[2] * 2, in_ch[2] * 2, stride=2)
            self.down3 = ConvLayer(out_ch[2], out_ch[2], stride=2)

        elif down_mode == 'maxpool':
            self.down1 = nn.MaxPool2d(2, 2)
            self.down2 = nn.MaxPool2d(2, 2)
            self.down3 = nn.MaxPool2d(2, 2)

    def forward(self, feats):
        x2_1 = self.EB2_1(concat_fusion(feats[1]))
        x3_1 = self.EB3_1(concat_fusion(feats[2]))
        x4_1 = self.EB4_1(concat_fusion(feats[3]))

        x3_2 = self.EB3_2(concat_fusion((feats[2][0], x3_1, self.down1(x2_1))))
        x4_2 = self.EB4_2(concat_fusion((feats[3][0], x4_1, self.down2(x3_1))))

        x4_3 = self.EB4_3(
            concat_fusion((feats[3][0], x4_1, x4_2, self.down3(x3_2))))

        return feats[0], x2_1, x3_2, x4_3


class LSDecoder(nn.Module):
    '''U-Net: Long Skip Connections'''
    def __init__(self, block, num_ch, up_mode='bilinear'):
        super(LSDecoder, self).__init__()
        self.DB1 = block(num_ch[0] + num_ch[1], num_ch[0])
        self.DB2 = block(num_ch[1] + num_ch[2], num_ch[1])
        self.DB3 = block(num_ch[2] + num_ch[3], num_ch[2])

        self.up = Upsample(up_mode, 2)

    def forward(self, feats):
        y3 = self.DB3(
            concat_fusion((feats[2], self.up(feats[3], feats[2].shape))))
        y2 = self.DB2(concat_fusion((feats[1], self.up(y3, feats[1].shape))))
        y1 = self.DB1(concat_fusion((feats[0], self.up(y2, feats[0].shape))))

        return y1


class NestDecoder(nn.Module):
    '''U-Net++: Nested Connections'''
    def __init__(self, block, num_ch, up_mode='bilinear'):
        super(NestDecoder, self).__init__()
        self.DB1_1 = block(num_ch[0] + num_ch[1], num_ch[0])
        self.DB2_1 = block(num_ch[1] + num_ch[2], num_ch[1])
        self.DB3_1 = block(num_ch[2] + num_ch[3], num_ch[2])

        self.DB1_2 = block(num_ch[0] * 2 + num_ch[1], num_ch[0])
        self.DB2_2 = block(num_ch[1] * 2 + num_ch[2], num_ch[1])

        self.DB1_3 = block(num_ch[0] * 3 + num_ch[1], num_ch[0])

        self.up = Upsample(up_mode, 2)

    def forward(self, feats):
        x1_1 = self.DB1_1(
            concat_fusion((feats[0], self.up(feats[1], feats[0].shape))))
        x2_1 = self.DB2_1(
            concat_fusion((feats[1], self.up(feats[2], feats[1].shape))))
        x3_1 = self.DB3_1(
            concat_fusion((feats[2], self.up(feats[3], feats[2].shape))))

        x1_2 = self.DB1_2(
            concat_fusion((feats[0], x1_1, self.up(x2_1, x1_1.shape))))
        x2_2 = self.DB2_2(
            concat_fusion((feats[1], x2_1, self.up(x3_1, x2_1.shape))))

        x1_3 = self.DB1_3(
            concat_fusion((feats[0], x1_1, x1_2, self.up(x2_2, x1_2.shape))))

        return x1_3


class FSDecoder(nn.Module):
    '''U-Net 3+: Full-scale Skip Connections'''
    def __init__(self, block, num_ch, up_mode='bilinear'):
        super(FSDecoder, self).__init__()
        # hid_ch = 32
        # cat_ch = hid_ch * 4
        cat_ch = num_ch[0] + num_ch[1] + num_ch[2] + num_ch[3]

        self.DB1 = block(cat_ch, num_ch[0])
        self.DB2 = block(cat_ch, num_ch[1])
        self.DB3 = block(cat_ch, num_ch[2])

        # self.down1_2 = ConvLayer(num_ch[0], hid_ch, ksize=1)
        # self.down1_3 = ConvLayer(num_ch[0], hid_ch, ksize=1)
        # self.down2_3 = ConvLayer(num_ch[1], hid_ch, ksize=1)

        # self.lateral1 = ConvLayer(num_ch[0], hid_ch, ksize=1)
        # self.lateral2 = ConvLayer(num_ch[1], hid_ch, ksize=1)
        # self.lateral3 = ConvLayer(num_ch[2], hid_ch, ksize=1)

        # self.up4_3 = ConvLayer(num_ch[3], hid_ch, ksize=1)
        # self.up4_2 = ConvLayer(num_ch[3], hid_ch, ksize=1)
        # self.up4_1 = ConvLayer(num_ch[3], hid_ch, ksize=1)
        # self.up3_2 = ConvLayer(num_ch[2], hid_ch, ksize=1)
        # self.up3_1 = ConvLayer(num_ch[2], hid_ch, ksize=1)
        # self.up2_1 = ConvLayer(num_ch[1], hid_ch, ksize=1)

        self.down1 = Downsample(2, 2)
        self.down2 = Downsample(4, 4)

        self.up1 = Upsample(up_mode, 2)
        self.up2 = Upsample(up_mode, 4)
        self.up3 = Upsample(up_mode, 8)

    def forward(self, feats):
        x1_3 = self.down2(feats[0], feats[2].shape)
        x2_3 = self.down1(feats[1], feats[2].shape)
        x4_3 = self.up1(feats[3], feats[2].shape)
        y3 = self.DB3(concat_fusion((x1_3, x2_3, feats[2], x4_3)))

        x1_2 = self.down1(feats[0], feats[1].shape)
        x3_2 = self.up1(y3, feats[1].shape)
        x4_2 = self.up2(feats[3], feats[1].shape)
        y2 = self.DB2(concat_fusion((x1_2, feats[1], x3_2, x4_2)))

        x2_1 = self.up1(y2, feats[0].shape)
        x3_1 = self.up2(y3, feats[0].shape)
        x4_1 = self.up3(feats[3], feats[0].shape)
        y1 = self.DB1(concat_fusion((feats[0], x2_1, x3_1, x4_1)))

        # x1_3 = self.down1_3(self.down2(feats[0], feats[2].shape))
        # x2_3 = self.down2_3(self.down1(feats[1], feats[2].shape))
        # x3_3 = self.lateral3(feats[2])
        # x4_3 = self.up4_3(self.up1(feats[3], feats[2].shape))
        # y3 = self.DB3(concat_fusion((x1_3, x2_3, x3_3, x4_3)))

        # x1_2 = self.down1_2(self.down1(feats[0], feats[1].shape))
        # x2_2 = self.lateral2(feats[1])
        # x3_2 = self.up3_2(self.up1(y3, feats[1].shape))
        # x4_2 = self.up4_2(self.up2(feats[3], feats[1].shape))
        # y2 = self.DB2(concat_fusion((x1_2, x2_2, x3_2, x4_2)))

        # x1_1 = self.lateral1(feats[0])
        # x2_1 = self.up2_1(self.up1(y2, feats[0].shape))
        # x3_1 = self.up3_1(self.up2(y3, feats[0].shape))
        # x4_1 = self.up4_1(self.up3(feats[3], feats[0].shape))
        # y1 = self.DB1(concat_fusion((x1_1, x2_1, x3_1, x4_1)))

        return y1


class Downsample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Downsample, self).__init__()
        self.down = nn.MaxPool2d(kernel_size, stride)

    def forward(self, feat, shape):
        out = self.down(feat)

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
