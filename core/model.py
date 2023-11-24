# -*- coding: utf-8 -*-
"""
# @file name  : model.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 网络模型类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .block import *
    from .fusion import *
except:
    from block import *
    from fusion import *

__all__ = [
    'PFNetv1', 'PFNetv2', 'DeepFuse', 'DenseFuse', 'VIFNet', 'DBNet',
    'SEDRFuse', 'NestFuse', 'RFNNest', 'UNFusion', 'Res2Fusion', 'MAFusion',
    'IFCNN', 'DIFNet', 'PMGI', 'MyFusion'
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

    def forward(self, img1, img2=None):
        if img2 is None:
            # extract
            feat = self.encoder(img1)

            # reconstruct
            recon_img = self.decoder(feat)

            return recon_img
        else:
            # extract
            feat1 = self.encoder(img1)
            feat2 = self.encoder(img2)

            # fuse
            fused_feat = self.fusion(feat1, feat2)

            # reconstruct
            fused_img = self.decoder(fused_feat)

            return fused_img


# 1. polarization and intensity image fusion


class PFNetv1(nn.Module):
    '''PFNet: An Unsupervised Deep Network for Polarization Image Fusion'''
    def __init__(self):
        super(PFNetv1, self).__init__()
        self.encode1 = nn.Sequential(
            ConvLayer(1, 16),
            DenseBlock(16, 16),
        )
        self.encode2 = nn.Sequential(
            ConvLayer(1, 16),
            DenseBlock(16, 16),
        )
        self.decode = nn.Sequential(
            ConvLayer(128, 128),
            ConvLayer(128, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, act=None),
        )

    def encoder(self, img1, img2):
        feat1 = self.encode1(img1)
        feat2 = self.encode2(img2)

        return feat1, feat2

    def fusion(self, feats):
        return concat_fusion(feats)

    def decoder(self, feat):
        return self.decode(feat)

    def forward(self, img1, img2):
        # extract
        feats = self.encoder(img1, img2)

        # fuse
        fused_feat = self.fusion(feats)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img


class PFNetv2(_FusionModel):
    '''Polarization Image Fusion with Self-Learned Fusion Strategy'''
    def __init__(self):
        super(PFNetv2, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 16),
            DenseBlock(16, 16),
        )
        self.fuse = nn.Sequential(
            ConvLayer(2, 2),
            ConvLayer(2, 2),
            ConvLayer(2, 1, act=None),
        )
        self.decode = nn.Sequential(
            ConvLayer(64, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, act=None),
        )

    def fusion(self, feat1, feat2):
        fused_feat = []

        for i in range(feat1.shape[1]):
            concat_feat = torch.stack((feat1[:, i], feat2[:, i]), dim=1)
            fused_feat.append(self.fuse(concat_feat))

        return concat_fusion(fused_feat) + feat1 + feat2


# 2. infrared and visible image fusion


class DeepFuse(_FusionModel):
    '''DeepFuse: A Deep Unsupervised Approach for Exposure Fusion with Extreme Exposure Image Pairs'''
    def __init__(self):
        super(DeepFuse, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 16, ksize=5),
            ConvLayer(16, 32, ksize=7),
        )
        self.decode = nn.Sequential(
            ConvLayer(32, 32, ksize=7),
            ConvLayer(32, 16, ksize=5),
            ConvLayer(16, 1, ksize=5, act=None),
        )

    def fusion(self, feat1, feat2, mode='sum'):
        return element_fusion(feat1, feat2, mode)


class DenseFuse(_FusionModel):
    '''DenseFuse: A Fusion Approach to Infrared and Visible Images'''
    def __init__(self):
        super(DenseFuse, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 16),
            DenseBlock(16, 16),
        )
        self.decode = nn.Sequential(
            ConvLayer(64, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, act=None),
        )

    def fusion(self, feat1, feat2, mode='sum'):
        if mode == 'sum':
            return element_fusion(feat1, feat2, mode)
        elif mode == 'l1':
            return attention_fusion(feat1, feat2, 'sa', spatial_mode=mode)
        else:
            raise ValueError("only supported ['sum', 'l1'] mode")


class VIFNet(_FusionModel):
    '''VIF-Net: An Unsupervised Framework for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(VIFNet, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 16),
            DenseBlock(16, 16),
        )
        self.decode = nn.Sequential(
            ConvLayer(128, 128),
            ConvLayer(128, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, act=None),
        )

    def fusion(self, feat1, feat2):
        return concat_fusion((feat1, feat2))


class DBNet(_FusionModel):
    '''A Dual-Branch Network for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(DBNet, self).__init__()
        self.encode = ConvLayer(1, 32)
        self.detail = nn.Sequential(
            ConvLayer(32, 16),
            DenseBlock(16, 16),
        )
        self.semantic = nn.Sequential(
            ConvLayer(32, 64, stride=2),
            ConvLayer(64, 128, stride=2),
            ConvLayer(128, 64, stride=2),
        )
        self.up = Upsample(mode='bilinear', scale_factor=8)
        self.decode = nn.Sequential(
            ConvLayer(128, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1, act=None),
        )

    def encoder(self, img):
        feat = self.encode(img)
        feat1 = self.detail(feat)
        feat2 = self.up(self.semantic(feat), feat.shape)

        return concat_fusion((feat1, feat2))

    def fusion(self, feat1, feat2, mode='sum'):
        if mode == 'sum':
            return element_fusion(feat1, feat2, mode)
        elif mode == 'avg':
            return attention_fusion(feat1, feat2, 'ca', channel_mode=mode)
        else:
            raise ValueError("only supported ['sum', 'avg'] mode")


class SEDRFuse(nn.Module):
    '''SEDRFuse: A Symmetric Encoder-Decoder with Residual Block Network for Infrared and Visible Image Fusion'''
    def __init__(self, norm=nn.GroupNorm):
        super(SEDRFuse, self).__init__()
        self.encode = nn.ModuleList([
            ConvLayer(1, 64, norm=norm),
            ConvLayer(64, 128, stride=2, norm=norm),
            ConvLayer(128, 256, stride=2, norm=norm),
            ResBlock(256, 256, norm1=norm, norm2=norm),
        ])
        self.decode = nn.ModuleList([
            ConvLayer(256,
                      128,
                      stride=2,
                      norm=norm,
                      layer=nn.ConvTranspose2d),
            ConvLayer(128,
                      64,
                      stride=2,
                      norm=norm,
                      layer=nn.ConvTranspose2d),
            ConvLayer(64, 1),
        ])

    def encoder(self, img):
        f_conv1 = self.encode[0](img)
        f_conv2 = self.encode[1](f_conv1)
        f_conv3 = self.encode[2](f_conv2)
        f_res = self.encode[3](f_conv3)

        return f_conv1, f_conv2, f_res

    def fusion(self, feat1, feat2):
        tmp1 = torch.abs(feat1)
        tmp2 = torch.abs(feat2)

        att1 = torch.softmax(tmp1, dim=1) * tmp1
        att2 = torch.softmax(tmp2, dim=1) * tmp2

        spatial1 = spatial_pooling(att1, mode='sum')
        spatial2 = spatial_pooling(att2, mode='sum')

        return weighted_fusion(feat1, feat2, spatial1, spatial2)

    def decoder(self, f_conv1, f_conv2, f_res):
        f_deconv1 = self.decode[0](f_res)
        f1 = F.relu(f_conv2 + f_deconv1)

        f_deconv2 = self.decode[1](f1)
        f2 = F.relu(f_conv1 + f_deconv2)

        out = self.decode[2](f2)

        return out

    def forward(self, img1, img2=None):
        if img2 is None:
            # extract
            f_conv1, f_conv2, f_res = self.encoder(img1)

            # reconstruct
            recon_img = self.decoder(f_conv1, f_conv2, f_res)

            return recon_img
        else:
            # extract
            f1_conv1, f1_conv2, f1_res = self.encoder(img1)
            f2_conv1, f2_conv2, f2_res = self.encoder(img2)

            # fuse
            f_conv1 = element_fusion(f1_conv1, f2_conv1, mode='max')
            f_conv2 = element_fusion(f1_conv2, f2_conv2, mode='max')
            f_res = self.fusion(f1_res, f2_res)

            # reconstruct
            fused_img = self.decoder(f_conv1, f_conv2, f_res)

            return fused_img


class NestFuse(_FusionModel):
    '''NestFuse: An Infrared and Visible Image Fusion Architecture Based on Nest Connection and Spatial/Channel Attention Models'''
    def __init__(self, down_mode='maxpool', up_mode='nearest'):
        super(NestFuse, self).__init__()
        num_ch = [64, 112, 160, 208]

        # encoder
        self.conv_in = ConvLayer(1, 16, ksize=1)
        self.CB1_0 = ConvBlock(16, num_ch[0])
        self.CB2_0 = ConvBlock(num_ch[0], num_ch[1])
        self.CB3_0 = ConvBlock(num_ch[1], num_ch[2])
        self.CB4_0 = ConvBlock(num_ch[2], num_ch[3])

        if down_mode == 'maxpool':
            self.down1 = nn.MaxPool2d(2, 2)
            self.down2 = nn.MaxPool2d(2, 2)
            self.down3 = nn.MaxPool2d(2, 2)
            
        elif down_mode == 'stride':
            self.down1 = ConvLayer(num_ch[0], num_ch[0], stride=2)
            self.down2 = ConvLayer(num_ch[1], num_ch[1], stride=2)
            self.down3 = ConvLayer(num_ch[2], num_ch[2], stride=2)

        # decoder
        self.decode = NestDecoder(ConvBlock, num_ch, up_mode)
        self.conv_out = ConvLayer(num_ch[0], 1, ksize=1)

    def encoder(self, img):
        x1_0 = self.CB1_0(self.conv_in(img))
        x2_0 = self.CB2_0(self.down1(x1_0))
        x3_0 = self.CB3_0(self.down2(x2_0))
        x4_0 = self.CB4_0(self.down3(x3_0))

        return x1_0, x2_0, x3_0, x4_0

    def fusion(self, feats1, feats2, mode='sca'):
        f1_0 = attention_fusion(feats1[0], feats2[0], mode)
        f2_0 = attention_fusion(feats1[1], feats2[1], mode)
        f3_0 = attention_fusion(feats1[2], feats2[2], mode)
        f4_0 = attention_fusion(feats1[3], feats2[3], mode)

        return f1_0, f2_0, f3_0, f4_0

    def decoder(self, feats):
        return self.conv_out(self.decode(feats))


class RFNNest(NestFuse):
    '''RFN-Nest: An End-to-End Residual Fusion Network for Infrared and Visible Images'''
    def __init__(self, down_mode='maxpool', up_mode='nearest'):
        super(RFNNest, self).__init__(down_mode, up_mode)
        num_ch = [64, 112, 160, 208]

        # fusion
        self.RFN1 = RFN(num_ch[0])
        self.RFN2 = RFN(num_ch[1])
        self.RFN3 = RFN(num_ch[2])
        self.RFN4 = RFN(num_ch[3])

    def fusion(self, feats1, feats2):
        f1_0 = self.RFN1(feats1[0], feats2[0])
        f2_0 = self.RFN2(feats1[1], feats2[1])
        f3_0 = self.RFN3(feats1[2], feats2[2])
        f4_0 = self.RFN4(feats1[3], feats2[3])

        return f1_0, f2_0, f3_0, f4_0


class UNFusion(_FusionModel):
    '''UNFusion: A Unified Multi-Scale Densely Connected Network for Infrared and Visible Image Fusion'''
    def __init__(self, down_mode='stride', up_mode='bilinear'):
        super(UNFusion, self).__init__()
        enc_ch = [16, 32, 48, 64]
        dec_ch = [16, 64, 256, 1024]

        # encoder
        self.CB1_0 = ConvLayer(1, enc_ch[0])
        self.CB2_0 = ConvLayer(enc_ch[0], enc_ch[1])
        self.CB3_0 = ConvLayer(enc_ch[1], enc_ch[2])
        self.CB4_0 = ConvLayer(enc_ch[2], enc_ch[3])

        if down_mode == 'maxpool':
            self.down1 = nn.MaxPool2d(2, 2)
            self.down2 = nn.MaxPool2d(2, 2)
            self.down3 = nn.MaxPool2d(2, 2)
            
        elif down_mode == 'stride':
            self.down1 = ConvLayer(enc_ch[0], enc_ch[0], stride=2)
            self.down2 = ConvLayer(enc_ch[1], enc_ch[1], stride=2)
            self.down3 = ConvLayer(enc_ch[2], enc_ch[2], stride=2)

        self.encode = NestEncoder(ECB, enc_ch, dec_ch, down_mode)

        # decoder
        self.decode = NestDecoder(DCB, dec_ch, up_mode)
        self.conv_out = ConvLayer(dec_ch[0], 1, ksize=1)

    def encoder(self, img):
        x1_0 = self.CB1_0(img)

        d1_0 = self.down1(x1_0)
        x2_0 = self.CB2_0(d1_0)

        d2_0 = self.down2(x2_0)
        x3_0 = self.CB3_0(d2_0)

        d3_0 = self.down3(x3_0)
        x4_0 = self.CB4_0(d3_0)

        return self.encode((x1_0, (x2_0, d1_0), (x3_0, d2_0), (x4_0, d3_0)))

    def fusion(self, feats1, feats2, mode='wavg'):
        f1_0 = attention_fusion(feats1[0], feats2[0], mode)
        f2_0 = attention_fusion(feats1[1], feats2[1], mode)
        f3_0 = attention_fusion(feats1[2], feats2[2], mode)
        f4_0 = attention_fusion(feats1[3], feats2[3], mode)

        return f1_0, f2_0, f3_0, f4_0

    def decoder(self, feats):
        return self.conv_out(self.decode(feats))


class Res2Fusion(_FusionModel):
    '''Res2Fusion: Infrared and Visible Image Fusion Based on Dense Res2net and Double Nonlocal Attention Models'''
    def __init__(self):
        super(Res2Fusion, self).__init__()
        # encoder
        self.conv_in = ConvLayer(1, 16)
        self.RB1 = Res2ConvBlock(16, 32, 4)
        self.RB2 = Res2ConvBlock(48, 64, 8)

        # decoder
        self.decode = nn.Sequential(
            ConvLayer(112, 64),
            ConvLayer(64, 32),
            ConvLayer(32, 16),
            ConvLayer(16, 1),
        )

    def encoder(self, img):
        x = self.conv_in(img)
        x = concat_fusion((x, self.RB1(x)))
        x = concat_fusion((x, self.RB2(x)))

        return x

    def fusion(self, feat1, feat2, mode='attn', spatial='nl', channel='nl'):
        if mode == 'elem':
            return element_fusion(feat1, feat2, 'mean')
        elif mode == 'attn':
            return attention_fusion(feat1, feat2, 'sca', spatial, channel)
        else:
            raise ValueError("only supported ['elem', 'attn'] mode")


class MAFusion(NestFuse):
    '''MAFusion: Multiscale Attention Network for Infrared and Visible Image Fusion'''
    def __init__(self, down_mode='maxpool', up_mode='bilinear'):
        super(MAFusion, self).__init__(down_mode, up_mode)
        num_ch = [64, 128, 256, 512]

        # encoder
        self.conv_in = ConvLayer(1, 16, ksize=1)
        self.CB1_0 = ConvBlock(16, num_ch[0])
        self.CB2_0 = ConvBlock(num_ch[0], num_ch[1])
        self.CB3_0 = ConvBlock(num_ch[1], num_ch[2])
        self.CB4_0 = ConvBlock(num_ch[2], num_ch[3])

        if down_mode == 'maxpool':
            self.down1 = nn.MaxPool2d(2, 2)
            self.down2 = nn.MaxPool2d(2, 2)
            self.down3 = nn.MaxPool2d(2, 2)
            
        elif down_mode == 'stride':
            self.down1 = ConvLayer(num_ch[0], num_ch[0], stride=2)
            self.down2 = ConvLayer(num_ch[1], num_ch[1], stride=2)
            self.down3 = ConvLayer(num_ch[2], num_ch[2], stride=2)

        # decoder
        self.decode = FSDecoder(ConvBlock, num_ch, up_mode)
        self.conv_out = ConvLayer(num_ch[0], 1, ksize=1)

    def fusion(self, feats1, feats2, mode='sca'):
        f1_0 = attention_fusion(feats1[0], feats2[0], mode)
        f2_0 = attention_fusion(feats1[1], feats2[1], mode)
        f3_0 = attention_fusion(feats1[2], feats2[2], mode)
        f4_0 = attention_fusion(feats1[3], feats2[3], mode)

        return f1_0, f2_0, f3_0, f4_0


# 3. general image fusion


class IFCNN(_FusionModel):
    '''IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network'''
    def __init__(self, norm=nn.BatchNorm2d):
        super(IFCNN, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 64, ksize=7, act=None),
            ConvLayer(64, 64, norm=norm),
        )
        self.decode = nn.Sequential(
            ConvLayer(64, 64, norm=norm),
            ConvLayer(64, 1, ksize=1, act=None),
        )

    def fusion(self, feat1, feat2, mode='max'):
        return element_fusion(feat1, feat2, mode)


class DIFNet(_FusionModel):
    '''Unsupervised Deep Image Fusion with Structure Tensor Representations'''
    def __init__(self, norm=nn.BatchNorm2d):
        super(DIFNet, self).__init__()
        self.encode = nn.Sequential(
            ConvLayer(1, 16),
            ResBlock(16, 16, norm1=norm),
            ResBlock(16, 16, norm1=norm),
        )
        self.fuse = ConvLayer(32, 16, act=None)
        self.decode = nn.Sequential(
            ResBlock(16, 16, norm1=norm),
            ResBlock(16, 16, norm1=norm),
            ResBlock(16, 16, norm1=norm),
            ConvLayer(16, 1, act=None),
        )

    def fusion(self, feat1, feat2):
        concat_feat = concat_fusion((feat1, feat2))
        fused_feat = self.fuse(concat_feat)

        return fused_feat


class PMGI(nn.Module):
    '''Rethinking the Image Fusion: A Fast Unified Image Fusion Network Based on Proportional Maintenance of Gradient and Intensity'''
    def __init__(self, norm=nn.BatchNorm2d, act=nn.LeakyReLU):
        super(PMGI, self).__init__()
        self.gradient = nn.ModuleList([
            ConvLayer(3, 16, ksize=5, norm=norm, act=act),
            ConvLayer(16, 16, norm=norm, act=act),
            ConvLayer(48, 16, norm=norm, act=act),
            ConvLayer(64, 16, norm=norm, act=act),
        ])
        self.intensity = nn.ModuleList([
            ConvLayer(3, 16, ksize=5, norm=norm, act=act),
            ConvLayer(16, 16, norm=norm, act=act),
            ConvLayer(48, 16, norm=norm, act=act),
            ConvLayer(64, 16, norm=norm, act=act),
        ])
        self.transfer1 = nn.ModuleList([
            ConvLayer(32, 16, ksize=1, norm=norm, act=act),
            ConvLayer(32, 16, ksize=1, norm=norm, act=act),
        ])
        self.transfer2 = nn.ModuleList([
            ConvLayer(32, 16, ksize=1, norm=norm, act=act),
            ConvLayer(32, 16, ksize=1, norm=norm, act=act),
        ])
        self.decode = ConvLayer(128, 1, ksize=1, act=nn.Tanh)

    def encoder(self, img1, img2):
        x1 = concat_fusion((img1, img1, img2))
        x2 = concat_fusion((img2, img2, img1))
        f0_1 = self.gradient[0](x1)
        f0_2 = self.intensity[0](x2)

        f1_1 = self.gradient[1](f0_1)
        f1_2 = self.intensity[1](f0_2)
        f1 = concat_fusion((f1_1, f1_2))
        f1_conv1 = self.transfer1[0](f1)
        f1_conv2 = self.transfer2[1](f1)
        f1_fuse1 = concat_fusion((f0_1, f1_1, f1_conv1))
        f1_fuse2 = concat_fusion((f0_2, f1_2, f1_conv2))

        f2_1 = self.gradient[2](f1_fuse1)
        f2_2 = self.intensity[2](f1_fuse2)
        f2 = concat_fusion((f2_1, f2_2))
        f2_conv1 = self.transfer2[0](f2)
        f2_conv2 = self.transfer2[1](f2)
        f2_fuse1 = concat_fusion((f0_1, f1_1, f2_1, f2_conv1))
        f2_fuse2 = concat_fusion((f0_2, f1_2, f2_2, f2_conv2))

        f3_1 = self.gradient[3](f2_fuse1)
        f3_2 = self.intensity[3](f2_fuse2)

        return f0_1, f0_2, f1_1, f1_2, f2_1, f2_2, f3_1, f3_2

    def fusion(self, feats):
        return concat_fusion(feats)

    def decoder(self, feat):
        return self.decode(feat)

    def forward(self, img1, img2):
        # extract
        feats = self.encoder(img1, img2)

        # fuse
        fused_feat = self.fusion(feats)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img / 2.0 + 0.5


# 4. my model


class MyFusion(nn.Module):
    def __init__(self,
                 encoder=SepConvBlock,
                 decoder=NestDecoder,
                 bias=False,
                 norm=None,
                 act=nn.ReLU6,
                 fusion_method='attn',
                 fusion_mode='sca',
                 down_mode='stride',
                 up_mode='bilinear',
                 share_weight_levels=4):
        super(MyFusion, self).__init__()
        num_ch = [16, 32, 64, 128]
        
        self.fusion_method = fusion_method
        self.fusion_mode = fusion_mode
        self.down_mode = down_mode
        self.up_mode = up_mode
        self.share_weight_levels = share_weight_levels

        # encoder
        self.conv_in_1 = ConvLayer(1,
                                   8,
                                   ksize=1,
                                   bias=bias,
                                   norm=norm,
                                   act=act)
        self.down1_1 = TransitionBlock(8,
                                       num_ch[0],
                                       stride=1,
                                       bias=bias,
                                       norm=norm,
                                       act=act)
        self.down2_1 = TransitionBlock(num_ch[0],
                                       num_ch[1],
                                       stride=2,
                                       bias=bias,
                                       norm=norm,
                                       act=act,
                                       down_mode=down_mode)
        self.down3_1 = TransitionBlock(num_ch[1],
                                       num_ch[2],
                                       stride=2,
                                       bias=bias,
                                       norm=norm,
                                       act=act,
                                       down_mode=down_mode)
        self.down4_1 = TransitionBlock(num_ch[2],
                                       num_ch[3],
                                       stride=2,
                                       bias=bias,
                                       norm=norm,
                                       act=act,
                                       down_mode=down_mode)

        if share_weight_levels < 4:
            self.conv_in_2 = ConvLayer(1,
                                       8,
                                       ksize=1,
                                       bias=bias,
                                       norm=norm,
                                       act=act)
            self.down1_2 = TransitionBlock(8,
                                           num_ch[0],
                                           stride=1,
                                           bias=bias,
                                           norm=norm,
                                           act=act)
        if share_weight_levels < 3:
            self.down2_2 = TransitionBlock(num_ch[0],
                                           num_ch[1],
                                           stride=2,
                                           bias=bias,
                                           norm=norm,
                                           act=act,
                                           down_mode=down_mode)
        if share_weight_levels < 2:
            self.down3_2 = TransitionBlock(num_ch[1],
                                           num_ch[2],
                                           stride=2,
                                           bias=bias,
                                           norm=norm,
                                           act=act,
                                           down_mode=down_mode)
        if share_weight_levels < 1:
            self.down4_2 = TransitionBlock(num_ch[2],
                                           num_ch[3],
                                           stride=2,
                                           bias=bias,
                                           norm=norm,
                                           act=act,
                                           down_mode=down_mode)

        if not isinstance(encoder, list):
            encoder = [encoder] * 4

        self.EB1_1 = encoder[0](num_ch[0], num_ch[0])
        self.EB2_1 = encoder[1](num_ch[1], num_ch[1])
        self.EB3_1 = encoder[2](num_ch[2], num_ch[2])
        self.EB4_1 = encoder[3](num_ch[3], num_ch[3])

        if share_weight_levels < 4:
            self.EB1_2 = encoder[0](num_ch[0], num_ch[0])
        if share_weight_levels < 3:
            self.EB2_2 = encoder[1](num_ch[1], num_ch[1])
        if share_weight_levels < 2:
            self.EB3_2 = encoder[2](num_ch[2], num_ch[2])
        if share_weight_levels < 1:
            self.EB4_2 = encoder[3](num_ch[3], num_ch[3])

        # fusion
        if fusion_method == 'elem':
            assert fusion_mode in ['sum', 'mean', 'max']
        elif fusion_method == 'attn':
            assert fusion_mode in ['sa', 'ca', 'sca']
        elif fusion_method == 'concat':
            self.fuse1 = ConvLayer(num_ch[0] * 2, num_ch[0], act=None)
            self.fuse2 = ConvLayer(num_ch[1] * 2, num_ch[1], act=None)
            self.fuse3 = ConvLayer(num_ch[2] * 2, num_ch[2], act=None)
            self.fuse4 = ConvLayer(num_ch[3] * 2, num_ch[3], act=None)
        elif fusion_method == 'rfn':
            self.RFN1 = RFN(num_ch[0])
            self.RFN2 = RFN(num_ch[1])
            self.RFN3 = RFN(num_ch[2])
            self.RFN4 = RFN(num_ch[3])

        # decoder
        self.decode = decoder(DCBlock, num_ch, up_mode)
        self.conv_out = ConvLayer(num_ch[0],
                                  1,
                                  ksize=1,
                                  bias=bias,
                                  norm=norm,
                                  act=act)

    def encoder(self, img1, img2):
        x1_1 = self.EB1_1(self.down1_1(self.conv_in_1(img1)))
        x2_1 = self.EB2_1(self.down2_1(x1_1))
        x3_1 = self.EB3_1(self.down3_1(x2_1))
        x4_1 = self.EB4_1(self.down4_1(x3_1))

        if self.share_weight_levels == 4:
            x1_2 = self.EB1_1(self.down1_1(self.conv_in_1(img2)))
            x2_2 = self.EB2_1(self.down2_1(x1_2))
            x3_2 = self.EB3_1(self.down3_1(x2_2))
            x4_2 = self.EB4_1(self.down4_1(x3_2))
        elif self.share_weight_levels == 3:
            x1_2 = self.EB1_2(self.down1_2(self.conv_in_2(img2)))
            x2_2 = self.EB2_1(self.down2_1(x1_2))
            x3_2 = self.EB3_1(self.down3_1(x2_2))
            x4_2 = self.EB4_1(self.down4_1(x3_2))
        elif self.share_weight_levels == 2:
            x1_2 = self.EB1_2(self.down1_2(self.conv_in_2(img2)))
            x2_2 = self.EB2_2(self.down2_2(x1_2))
            x3_2 = self.EB3_1(self.down3_1(x2_2))
            x4_2 = self.EB4_1(self.down4_1(x3_2))
        elif self.share_weight_levels == 1:
            x1_2 = self.EB1_2(self.down1_2(self.conv_in_2(img2)))
            x2_2 = self.EB2_2(self.down2_2(x1_2))
            x3_2 = self.EB3_2(self.down3_2(x2_2))
            x4_2 = self.EB4_1(self.down4_1(x3_2))
        elif self.share_weight_levels == 0:
            x1_2 = self.EB1_2(self.down1_2(self.conv_in_2(img2)))
            x2_2 = self.EB2_2(self.down2_2(x1_2))
            x3_2 = self.EB3_2(self.down3_2(x2_2))
            x4_2 = self.EB4_2(self.down4_2(x3_2))

        return (x1_1, x2_1, x3_1, x4_1), (x1_2, x2_2, x3_2, x4_2)

    def fusion(self, feats1, feats2):
        if self.fusion_method == 'elem':
            f1 = element_fusion(feats1[0], feats2[0], self.fusion_mode)
            f2 = element_fusion(feats1[1], feats2[1], self.fusion_mode)
            f3 = element_fusion(feats1[2], feats2[2], self.fusion_mode)
            f4 = element_fusion(feats1[3], feats2[3], self.fusion_mode)
        elif self.fusion_method == 'attn':
            f1 = attention_fusion(feats1[0], feats2[0], self.fusion_mode)
            f2 = attention_fusion(feats1[1], feats2[1], self.fusion_mode)
            f3 = attention_fusion(feats1[2], feats2[2], self.fusion_mode)
            f4 = attention_fusion(feats1[3], feats2[3], self.fusion_mode)
        elif self.fusion_method == 'concat':
            f1 = self.fuse1(concat_fusion((feats1[0], feats2[0])))
            f2 = self.fuse2(concat_fusion((feats1[1], feats2[1])))
            f3 = self.fuse3(concat_fusion((feats1[2], feats2[2])))
            f4 = self.fuse4(concat_fusion((feats1[3], feats2[3])))
        elif self.fusion_method == 'rfn':
            f1 = self.RFN1(feats1[0], feats2[0])
            f2 = self.RFN2(feats1[1], feats2[1])
            f3 = self.RFN3(feats1[2], feats2[2])
            f4 = self.RFN4(feats1[3], feats2[3])
        else:
            raise ValueError("only supported ['elem', 'attn', 'concat', 'rfn'] method")

        return f1, f2, f3, f4

    def decoder(self, feats):
        out = self.conv_out(self.decode(feats))

        return out

    def forward(self, img1, img2):
        # extract
        feats1, feats2 = self.encoder(img1, img2)

        # fuse
        fused_feat = self.fusion(feats1, feats2)

        # reconstruct
        fused_img = self.decoder(fused_feat)

        return fused_img


if __name__ == '__main__':

    import time
    from thop import clever_format, profile

    # models = [
    #     DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse, NestFuse, RFNNest,
    #     UNFusion, Res2Fusion, MAFusion, IFCNN, DIFNet, PMGI, PFNetv1, PFNetv2
    # ]

    # model = models[0]
    # print(f'model: {model.__name__}')

    # model = model()

    encoders = [
        SepConvBlock, MixConvBlock, Res2ConvBlock, ConvFormerBlock,
        MixFormerBlock, Res2FormerBlock, TransformerBlock
    ]
    decoders = [LSDecoder, NestDecoder, FSDecoder]

    # encoder = [encoders[-2], encoders[-2], encoders[-1], encoders[-1]]
    encoder = encoders[0]
    decoder = decoders[0]

    if not isinstance(encoder, list):
        print(f'encoder: {encoder.__name__}')
    else:
        [print(f'encoder{i + 1}: {e.__name__}') for i, e in enumerate(encoder)]

    print(f'decoder: {decoder.__name__}')

    model = MyFusion(encoder, decoder, share_weight_levels=4)

    params = sum([param.numel() for param in model.parameters()])
    print(f'params: {params / 1e6:.3f}M')

    # print(model)

    x1 = torch.rand(1, 1, 256, 256)
    x2 = torch.rand(1, 1, 256, 256)

    flops, params = profile(model, inputs=(x1, x2))
    flops, params = clever_format([flops, params], '%.3f')
    print(f'params: {params}, flops: {flops}')

    # # out = model(x1)
    # out = model(x1, x2)
    # print(out.shape)

    t = 0.0
    for i in range(51):
        if i > 0:
            t0 = time.time()

        with torch.no_grad():
            out = model(x1, x2)

        if i > 0:
            t += time.time() - t0
    print(f'time: {t / 50 * 1000:.3f}ms')
