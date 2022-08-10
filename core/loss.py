# -*- coding: utf-8 -*-
"""
# @file name  : loss.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 损失函数类
"""

from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / (2.0 * sigma**2))
        for x in range(window_size)
    ])

    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()

    return window


def _ssim(img1,
          img2,
          window=None,
          window_size=11,
          val_range=None,
          size_average=False,
          full=True):
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

    c, h, w = img1.shape[1:]

    if window is None:
        min_size = min(window_size, h, w)
        window = create_window(min_size, channel=c).to(img1)

    mu1 = F.conv2d(img1, window, padding=0, groups=c)
    mu2 = F.conv2d(img2, window, padding=0, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=c) - mu1_mu2

    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    m1 = 2.0 * mu1_mu2 + C1
    m2 = mu1_sq + mu2_sq + C1
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    if full:
        ssim_map = (m1 / m2) * (v1 / v2)
    else:
        ssim_map = v1 / v2

    v = torch.full_like(sigma1_sq, 1e-4)
    sigma1 = torch.where(sigma1_sq < 1e-4, v, sigma1_sq)

    if size_average:
        return ssim_map.mean(), sigma1.mean()
    else:
        return ssim_map, sigma1


class SSIM(nn.Module):
    def __init__(self,
                 window_size=11,
                 val_range=None,
                 size_average=False,
                 full=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.val_range = val_range
        self.size_average = size_average
        self.full = full

    def forward(self, img, pred):
        channel = img.shape[1]
        window = create_window(self.window_size, channel).to(img)

        return _ssim(img,
                     pred,
                     window=window,
                     window_size=self.window_size,
                     val_range=self.val_range,
                     size_average=self.size_average,
                     full=self.full)


class MSWSSIM(nn.Module):
    def __init__(self,
                 window_sizes=(11, 9, 7, 5, 3),
                 val_range=None,
                 size_average=False,
                 full=True):
        super(MSWSSIM, self).__init__()
        self.window_sizes = window_sizes
        self.val_range = val_range
        self.size_average = size_average
        self.full = full
        self.loss_fns = (SSIM(s, val_range, size_average, full)
                         for s in window_sizes)

    def forward(self, img1, img2, pred):
        ssim = 0.0

        for loss_fn in self.loss_fns:
            ssim1, sigma1 = loss_fn(img1, pred)
            ssim2, sigma2 = loss_fn(img2, pred)
            gamma = sigma1 / (sigma1 + sigma2).clamp_(min=1e-7)
            ssim += torch.mean(gamma * ssim1) + torch.mean(
                (1.0 - gamma) * ssim2)

        return ssim / len(self.window_sizes)


class FusionLoss(nn.Module):
    def __init__(self,
                 mode='ssim',
                 extra_mode=None,
                 val_range=None,
                 size_average=False,
                 full=True):
        super(FusionLoss, self).__init__()
        self.mode = mode
        self.extra_mode = extra_mode
        self.val_range = val_range
        self.size_average = size_average
        self.full = full

    def forward(self, img1, img2, pred):
        if self.mode == 'ssim':
            loss_fn = SSIM(11, self.val_range, self.size_average, self.full)
            ssim1 = loss_fn(img1, pred)[0]
            ssim2 = loss_fn(img2, pred)[0]
            loss = 1.0 - (torch.mean(ssim1) + torch.mean(ssim2)) * 0.5

        elif self.mode == 'mswssim':
            loss_fn = MSWSSIM((11, 9, 7, 5, 3), self.val_range,
                              self.size_average, self.full)
            loss = 1.0 - loss_fn(img1, img2, pred)

        else:
            raise ValueError("only supported ['ssim', 'mswssim'] mode")

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

    mode = ['ssim', 'mswssim']
    extra_mode = ['l1', 'l2']

    model = FusionLoss(mode[1],
                       extra_mode[0],
                       val_range=1,
                       size_average=False,
                       full=True)

    x1 = torch.rand(2, 1, 224, 224)
    x2 = torch.rand(2, 1, 224, 224)
    y = torch.rand(2, 1, 224, 224)

    loss1, loss2 = model(x1, x2, y)
    total_loss = loss1 + loss2 * 0.1
    print(total_loss.item())
