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
    'SEDRFuse', 'NestFuse', 'RFNNest', 'UNFusion', 'IFCNN', 'DIFNet', 'PMGI'
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

    def forward(self, img1, img2=None, mode=None):
        if mode == 'ae':
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
            ConvBlock(16, 1, act=None),
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
            ConvBlock(1, 16),
            DenseBlock(3, 16, 16),
        )
        self.fuse = nn.Sequential(
            ConvBlock(2, 2),
            ConvBlock(2, 2),
            ConvBlock(2, 1, act=None),
        )
        self.decode = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, act=None),
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
            ConvBlock(1, 16, ksize=5),
            ConvBlock(16, 32, ksize=7),
        )
        self.decode = nn.Sequential(
            ConvBlock(32, 32, ksize=7),
            ConvBlock(32, 16, ksize=5),
            ConvBlock(16, 1, ksize=5, act=None),
        )

    def fusion(self, feat1, feat2, mode='sum'):
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
            ConvBlock(16, 1, act=None),
        )

    def fusion(self, feat1, feat2, mode='sum'):
        if mode == 'sum':
            return element_fusion(feat1, feat2, mode)
        elif mode == 'l1':
            return spatial_fusion(feat1, feat2, mode, softmax=False)
        else:
            raise ValueError("only supported ['sum', 'l1'] mode")


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
            ConvBlock(16, 1, act=None),
        )

    def fusion(self, feat1, feat2):
        return concat_fusion((feat1, feat2))


class DBNet(_FusionModel):
    '''A Dual-Branch Network for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(DBNet, self).__init__()
        self.encode = ConvBlock(1, 32)
        self.detail = nn.Sequential(
            ConvBlock(32, 16),
            DenseBlock(3, 16, 16),
        )
        self.semantic = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 64, stride=2),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.decode = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1, act=None),
        )

    def encoder(self, img):
        feat = self.encode(img)
        feat1 = self.detail(feat)
        feat2 = self.semantic(feat)

        return concat_fusion((feat1, feat2))

    def fusion(self, feat1, feat2, mode='sum'):
        if mode == 'sum':
            return element_fusion(feat1, feat2, mode)
        elif mode == 'avg':
            return channel_fusion(feat1, feat2, mode, softmax=False)
        else:
            raise ValueError("only supported ['sum', 'avg'] mode")


class SEDRFuse(nn.Module):
    '''SEDRFuse: A Symmetric Encoder-Decoder with Residual Block Network for Infrared and Visible Image Fusion'''
    def __init__(self):
        super(SEDRFuse, self).__init__()
        self.encode = nn.ModuleList([
            ConvBlock(1, 64, norm='in'),
            ConvBlock(64, 128, stride=2, norm='in'),
            ConvBlock(128, 256, stride=2, norm='in'),
            ResBlock(256, 256, norm1='in', norm2='in'),
        ])
        self.decode = nn.ModuleList([
            DeconvBlock(256, 128, stride=2, output_padding=1, norm='in'),
            DeconvBlock(128, 64, stride=2, output_padding=1, norm='in'),
            ConvBlock(64, 1),
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

    def forward(self, img1, img2=None, mode=None):
        if mode == 'ae':
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
    def __init__(self):
        super(NestFuse, self).__init__()
        num_ch = [64, 112, 160, 208]

        # encoder
        self.conv_in = ConvBlock(1, 16, ksize=1)
        self.CB1_0 = CB(16, num_ch[0])
        self.CB2_0 = CB(num_ch[0], num_ch[1])
        self.CB3_0 = CB(num_ch[1], num_ch[2])
        self.CB4_0 = CB(num_ch[2], num_ch[3])
        self.down = nn.MaxPool2d(2, 2)

        # decoder
        self.decode = NestDecoder(CB, num_ch)
        self.conv_out = ConvBlock(num_ch[0], 1)

    def encoder(self, img):
        x1_0 = self.CB1_0(self.conv_in(img))
        x2_0 = self.CB2_0(self.down(x1_0))
        x3_0 = self.CB3_0(self.down(x2_0))
        x4_0 = self.CB4_0(self.down(x3_0))

        return x1_0, x2_0, x3_0, x4_0

    def fusion(self, feats1, feats2, mode='mean'):
        f1_0 = attention_fusion(feats1[0], feats2[0], mode)
        f2_0 = attention_fusion(feats1[1], feats2[1], mode)
        f3_0 = attention_fusion(feats1[2], feats2[2], mode)
        f4_0 = attention_fusion(feats1[3], feats2[3], mode)

        return f1_0, f2_0, f3_0, f4_0

    def decoder(self, feats):
        return self.conv_out(self.decode(feats))


class RFNNest(NestFuse):
    '''RFN-Nest: An End-to-End Residual Fusion Network for Infrared and Visible Images'''
    def __init__(self):
        super(RFNNest, self).__init__()
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
        self.CB1_0 = ConvBlock(1, enc_ch[0])
        self.CB2_0 = ConvBlock(enc_ch[0], enc_ch[1])
        self.CB3_0 = ConvBlock(enc_ch[1], enc_ch[2])
        self.CB4_0 = ConvBlock(enc_ch[2], enc_ch[3])

        self.down1 = ConvBlock(
            enc_ch[0], enc_ch[0],
            stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)
        self.down2 = ConvBlock(
            enc_ch[1], enc_ch[1],
            stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)
        self.down3 = ConvBlock(
            enc_ch[2], enc_ch[2],
            stride=2) if down_mode == 'stride' else nn.MaxPool2d(2, 2)

        self.encode = NestEncoder(ECB, enc_ch, dec_ch, down_mode)

        # decoder
        self.decode = NestDecoder(DCB, dec_ch, up_mode)
        self.conv_out = ConvBlock(dec_ch[0], 1)

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


# 3. general image fusion


class IFCNN(_FusionModel):
    '''IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network'''
    def __init__(self):
        super(IFCNN, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 64, ksize=7, act=None),
            ConvBlock(64, 64, norm='bn'),
        )
        self.decode = nn.Sequential(
            ConvBlock(64, 64, norm='bn'),
            ConvBlock(64, 1, ksize=1, act=None),
        )

    def fusion(self, feat1, feat2, mode='max'):
        return element_fusion(feat1, feat2, mode)


class DIFNet(_FusionModel):
    '''Unsupervised Deep Image Fusion with Structure Tensor Representations'''
    def __init__(self):
        super(DIFNet, self).__init__()
        self.encode = nn.Sequential(
            ConvBlock(1, 16),
            ResBlock(16, 16, norm1='bn'),
            ResBlock(16, 16, norm1='bn'),
        )
        self.fuse = ConvBlock(32, 16, act=None)
        self.decode = nn.Sequential(
            ResBlock(16, 16, norm1='bn'),
            ResBlock(16, 16, norm1='bn'),
            ResBlock(16, 16, norm1='bn'),
            ConvBlock(16, 1, act=None),
        )

    def fusion(self, feat1, feat2):
        concat_feat = concat_fusion((feat1, feat2))
        fused_feat = self.fuse(concat_feat)

        return fused_feat


class PMGI(nn.Module):
    '''Rethinking the Image Fusion: A Fast Unified Image Fusion Network Based on Proportional Maintenance of Gradient and Intensity'''
    def __init__(self):
        super(PMGI, self).__init__()
        self.gradient = nn.ModuleList([
            ConvBlock(3, 16, ksize=5, norm='bn', act='lrelu'),
            ConvBlock(16, 16, norm='bn', act='lrelu'),
            ConvBlock(48, 16, norm='bn', act='lrelu'),
            ConvBlock(64, 16, norm='bn', act='lrelu'),
        ])
        self.intensity = nn.ModuleList([
            ConvBlock(3, 16, ksize=5, norm='bn', act='lrelu'),
            ConvBlock(16, 16, norm='bn', act='lrelu'),
            ConvBlock(48, 16, norm='bn', act='lrelu'),
            ConvBlock(64, 16, norm='bn', act='lrelu'),
        ])
        self.transfer1 = nn.ModuleList([
            ConvBlock(32, 16, ksize=1, norm='bn', act='lrelu'),
            ConvBlock(32, 16, ksize=1, norm='bn', act='lrelu'),
        ])
        self.transfer2 = nn.ModuleList([
            ConvBlock(32, 16, ksize=1, norm='bn', act='lrelu'),
            ConvBlock(32, 16, ksize=1, norm='bn', act='lrelu'),
        ])
        self.decode = ConvBlock(128, 1, ksize=1, act='tanh')

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

        return fused_img


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    models = (PFNetv1, PFNetv2, DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse,
              NestFuse, RFNNest, UNFusion, IFCNN, DIFNet, PMGI)

    model = models[0]()
    print(model)
    # summary(model, [(1, 224, 224), (1, 224, 224)], 2, device='cpu')

    x1 = torch.rand(2, 1, 224, 224)
    x2 = torch.rand(2, 1, 224, 224)

    # out = model(x1, mode='ae')
    out = model(x1, x2)
    print(out.shape)
