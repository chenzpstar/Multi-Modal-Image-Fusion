# -*- coding: utf-8 -*-
"""
# @file name  : loss.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 损失函数类
# @reference  : https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


def _gaussian_kernel(window_size, sigma=1.5):
    gauss = torch.FloatTensor([
        exp(-(x - window_size // 2)**2 / (2.0 * sigma**2))
        for x in range(window_size)
    ])

    return gauss / gauss.sum()


def _create_window(window_size, sigma=1.5, channel=1):
    _1D_window = _gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2D_window = torch.mm(_1D_window, _1D_window.t()).unsqueeze(0).unsqueeze(0)

    return _2D_window.repeat(channel, 1, 1, 1)


def _gaussian_filter(img, window, use_pad=False):
    channel = img.shape[1]
    padding = window.shape[-1] // 2 if use_pad else 0

    return F.conv2d(img, window, padding=padding, groups=channel)


def _ssim(img1,
          img2,
          window=None,
          window_size=11,
          val_range=None,
          size_average=True,
          no_luminance=False):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0

        L = max_val - min_val
    else:
        L = val_range

    if window is None:
        c, h, w = img1.shape[1:]
        min_size = min(window_size, h, w)
        window = _create_window(min_size, channel=c)

    window = window.to(img1)

    mu1 = _gaussian_filter(img1, window)
    mu2 = _gaussian_filter(img2, window)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter(img1 * img1, window) - mu1_sq
    sigma2_sq = _gaussian_filter(img2 * img2, window) - mu2_sq
    sigma12 = _gaussian_filter(img1 * img2, window) - mu1_mu2

    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L)**2, (K2 * L)**2

    m1 = 2.0 * mu1_mu2 + C1
    m2 = mu1_sq + mu2_sq + C1
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    l = m1 / m2 if not no_luminance else 1.0
    cs = v1 / v2
    ssim = l * cs

    v = torch.full_like(sigma1_sq, 1e-4)
    sigma1 = torch.where(sigma1_sq < 1e-4, v, sigma1_sq)

    if size_average:
        ssim, cs, sigma1 = ssim.mean(), cs.mean(), sigma1.mean()
    else:
        ssim, cs, sigma1 = ssim.mean(dim=(1, 2, 3)), cs.mean(
            dim=(1, 2, 3)), sigma1.mean(dim=(1, 2, 3))

    out = {'ssim': ssim, 'cs': cs, 'sigma': sigma1}

    return out


def _msssim(img1,
            img2,
            window=None,
            weights=None,
            window_size=11,
            val_range=None,
            size_average=True,
            no_luminance=False):
    if window is None:
        c, h, w = img1.shape[1:]
        min_size = min(window_size, h, w)
        window = _create_window(min_size, channel=c)

    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    window = window.to(img1)
    weights = weights.to(img1)

    values = []
    levels = len(weights)

    for i in range(levels):
        out = _ssim(img1, img2, window, window_size, val_range, size_average,
                    no_luminance)

        if i < levels - 1:
            values.append(out['cs'])
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
        else:
            values.append(out['ssim'])

    values = torch.stack(values, dim=0).clamp(min=eps)

    pow1 = values[:-1]**weights[:-1]
    pow2 = values[-1:]**weights[-1:]
    msssim = torch.prod(pow1[:-1] * pow2[-1], dim=0)

    return msssim


class SSIM(nn.Module):
    '''Structural Similarity Index'''
    def __init__(self,
                 window_size=11,
                 channel=1,
                 val_range=None,
                 size_average=True,
                 no_luminance=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.val_range = val_range
        self.size_average = size_average
        self.no_luminance = no_luminance

        window = _create_window(window_size, channel)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        return _ssim(img1,
                     img2,
                     self.window,
                     val_range=self.val_range,
                     size_average=self.size_average,
                     no_luminance=self.no_luminance)


class MS_SSIM(SSIM):
    '''Multi-Scale Structural Similarity Index'''
    def __init__(self,
                 window_size=11,
                 channel=1,
                 val_range=None,
                 size_average=True,
                 no_luminance=False):
        super(MS_SSIM, self).__init__(window_size, channel, val_range,
                                      size_average, no_luminance)

        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.register_buffer('weights', weights)

    def forward(self, img1, img2):
        return _msssim(img1,
                       img2,
                       self.window,
                       self.weights,
                       val_range=self.val_range,
                       size_average=self.size_average,
                       no_luminance=self.no_luminance)


class MSW_SSIM(nn.Module):
    '''Multi-Scale and Weighted Structural Similarity Index'''
    def __init__(self,
                 window_sizes=(11, 9, 7, 5, 3),
                 channel=1,
                 val_range=None,
                 size_average=True,
                 no_luminance=False):
        super(MSW_SSIM, self).__init__()
        self.window_sizes = window_sizes
        self.channel = channel
        self.val_range = val_range
        self.size_average = size_average
        self.no_luminance = no_luminance
        self.ssim_fns = (SSIM(s, channel, val_range, size_average,
                              no_luminance) for s in window_sizes)

    def forward(self, img1, img2, pred):
        ssim = 0.0

        for ssim_fn in self.ssim_fns:
            out1 = ssim_fn(img1, pred)
            out2 = ssim_fn(img2, pred)
            gamma = out1['sigma'] / (out1['sigma'] +
                                     out2['sigma']).clamp_(min=eps)
            ssim += torch.mean(gamma * out1['ssim']) + torch.mean(
                (1.0 - gamma) * out2['ssim'])

        return ssim / len(self.window_sizes)


class FusionLoss(nn.Module):
    def __init__(self,
                 mode='ssim',
                 extra_mode=None,
                 val_range=None,
                 size_average=True,
                 no_luminance=False):
        super(FusionLoss, self).__init__()
        self.mode = mode
        self.extra_mode = extra_mode
        self.val_range = val_range
        self.size_average = size_average
        self.no_luminance = no_luminance

    def forward(self, img1, img2, pred):
        channel = img1.shape[1]

        if self.mode == 'ssim':
            ssim_fn = SSIM(11, channel, self.val_range, self.size_average,
                           self.no_luminance)
            ssim1 = ssim_fn(img1, pred)['ssim']
            ssim2 = ssim_fn(img2, pred)['ssim']
            loss = 1.0 - (torch.mean(ssim1) + torch.mean(ssim2)) * 0.5

        elif self.mode == 'ms-ssim':
            msssim_fn = MS_SSIM(11, channel, self.val_range, self.size_average,
                                self.no_luminance)
            msssim1 = msssim_fn(img1, pred)
            msssim2 = msssim_fn(img2, pred)
            loss = 1.0 - (torch.mean(msssim1) + torch.mean(msssim2)) * 0.5

        elif self.mode == 'msw-ssim':
            mswssim_fn = MSW_SSIM((11, 9, 7, 5, 3), channel, self.val_range,
                                  self.size_average, self.no_luminance)
            loss = 1.0 - mswssim_fn(img1, img2, pred)

        else:
            raise ValueError(
                "only supported ['ssim', 'ms-ssim', 'msw-ssim'] mode")

        if self.extra_mode is None:
            extra_loss = 0.0

        elif self.extra_mode == 'l1':
            avg = (img1 + img2) / 2.0
            extra_loss = torch.mean(torch.abs(avg - pred))

        elif self.extra_mode == 'l2':
            avg = (img1 + img2) / 2.0
            extra_loss = torch.mean((avg - pred).pow(2))

        else:
            raise ValueError("only supported ['l1', 'l2'] mode")

        return loss, extra_loss


if __name__ == '__main__':

    import torch

    torch.manual_seed(0)

    mode = ['ssim', 'ms-ssim', 'msw-ssim']
    extra_mode = ['l1', 'l2']

    model = FusionLoss(mode[2],
                       extra_mode[0],
                       val_range=1,
                       size_average=True,
                       no_luminance=False)

    x1 = torch.rand(2, 1, 224, 224)
    x2 = torch.rand(2, 1, 224, 224)
    y = torch.rand(2, 1, 224, 224)

    loss1, loss2 = model(x1, x2, y)
    print(loss1.item()), print(loss2.item())

    total_loss = loss1 + loss2 * 0.1
    print(total_loss.item())
