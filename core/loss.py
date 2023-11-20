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

__all__ = [
    'SSIM', 'MS_SSIM', 'MSW_SSIM', 'SSIMLoss', 'PixelLoss', 'GradLoss',
    'TVLoss', 'NormLoss'
]

eps = 1e-7


def _gaussian_kernel(win_size, sigma):
    gauss = torch.FloatTensor([
        exp(-(x - win_size // 2)**2 / (2.0 * sigma**2))
        for x in range(win_size)
    ])

    return gauss / gauss.sum()


def create_window(win_size):
    sigma = 1.5 if win_size == 11 else 0.15 * (win_size - 1)

    _1D_window = _gaussian_kernel(win_size, sigma).unsqueeze(1)
    _2D_window = torch.mm(_1D_window, _1D_window.t())

    return _2D_window.unsqueeze(0).unsqueeze(0)


def _gaussian_fn(img, window, use_padding=False):
    channel = img.shape[1]

    if use_padding:
        p = window.shape[-1] // 2
        img = F.pad(img, (p, p, p, p), 'reflect')

    return F.conv2d(img, window, groups=channel)


def cmpt_ssim(img1,
              img2,
              win_size=11,
              window=None,
              data_range=None,
              use_padding=False,
              size_average=True):
    # Data range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if data_range is None:
        max_val = 255.0 if img1.max() > 128 else 1.0
        min_val = -1.0 if img1.min() < -0.5 else 0.0
        L = max_val - min_val
    else:
        L = data_range

    if window is None:
        h, w = img1.shape[-2:]
        min_size = min(win_size, h, w)
        window = create_window(min_size)
        window = nn.Parameter(window, requires_grad=False)

    window = window.to(img1, non_blocking=True)

    im1 = img1.clone()
    im2 = img2.clone()

    mu1 = _gaussian_fn(im1, window, use_padding)
    mu2 = _gaussian_fn(im2, window, use_padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (_gaussian_fn(im1 * im1, window, use_padding) -
                 mu1_sq).clamp(min=0)
    sigma2_sq = (_gaussian_fn(im2 * im2, window, use_padding) -
                 mu2_sq).clamp(min=0)
    sigma12 = _gaussian_fn(im1 * im2, window, use_padding) - mu1_mu2

    K1, K2 = 0.01, 0.03
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    m1 = 2.0 * mu1_mu2 + C1
    m2 = mu1_sq + mu2_sq + C1
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    cs = v1 / v2
    ssim = (m1 * v1) / (m2 * v2)

    sigma = sigma1_sq.clamp(min=1e-4)

    if size_average:
        ssim = ssim.mean(dim=(1, 2, 3))
        cs = cs.mean(dim=(1, 2, 3))
        sigma = sigma.mean(dim=(1, 2, 3))

    return {'ssim': ssim, 'cs': cs, 'sigma': sigma}


def cmpt_msssim(img1,
                img2,
                win_size=11,
                window=None,
                weights=None,
                data_range=None,
                use_padding=False,
                size_average=True):
    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        weights = nn.Parameter(weights, requires_grad=False)

    if window is None:
        h, w = img1.shape[-2:]
        min_size = min(win_size, h, w)
        window = create_window(min_size)
        window = nn.Parameter(window, requires_grad=False)

    weights = weights.to(img1, non_blocking=True)
    window = window.to(img1, non_blocking=True)

    im1 = img1.clone()
    im2 = img2.clone()

    values = []
    levels = len(weights)

    for i in range(levels):
        out = cmpt_ssim(im1, im2, win_size, window, data_range, use_padding,
                        size_average)

        if i < levels - 1:
            values.append(out['cs'])

            h, w = im1.shape[-2:]
            pad_h, pad_w = h % 2, w % 2
            im1 = F.pad(im1, (0, pad_w, 0, pad_h), 'reflect')
            im2 = F.pad(im2, (0, pad_w, 0, pad_h), 'reflect')

            im1 = F.avg_pool2d(im1, 2, 2)
            im2 = F.avg_pool2d(im2, 2, 2)
        else:
            values.append(out['ssim'])

    values = torch.stack(values, dim=0).clamp(min=eps)
    msssim = torch.prod(values**weights.unsqueeze(1), dim=0)

    return msssim


class SSIM(nn.Module):
    '''Structural Similarity Index'''
    def __init__(self,
                 win_size=11,
                 data_range=1.0,
                 use_padding=False,
                 size_average=True):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.data_range = data_range
        self.use_padding = use_padding
        self.size_average = size_average

        window = create_window(win_size)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        return cmpt_ssim(img1,
                         img2,
                         window=self.window,
                         data_range=self.data_range,
                         use_padding=self.use_padding,
                         size_average=self.size_average)


class MS_SSIM(SSIM):
    '''Multi-Scale Structural Similarity Index'''
    def __init__(self,
                 win_size=11,
                 data_range=1.0,
                 use_padding=False,
                 size_average=True):
        super(MS_SSIM, self).__init__(win_size, data_range, use_padding,
                                      size_average)

        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        self.register_buffer('weights', weights)

    def forward(self, img1, img2):
        return cmpt_msssim(img1,
                           img2,
                           window=self.window,
                           weights=self.weights,
                           data_range=self.data_range,
                           use_padding=self.use_padding,
                           size_average=self.size_average)


class MSW_SSIM(nn.Module):
    '''Multi-Scale and Weighted Structural Similarity Index'''
    def __init__(self,
                 win_sizes=(11, 9, 7, 5, 3),
                 data_range=1.0,
                 use_padding=False,
                 size_average=False):
        super(MSW_SSIM, self).__init__()
        self.win_sizes = win_sizes
        self.data_range = data_range
        self.use_padding = use_padding
        self.size_average = size_average
        self.ssim_fns = (SSIM(win_size, data_range, use_padding, size_average)
                         for win_size in win_sizes)

    def forward(self, img1, img2, imgf):
        ssim = 0.0

        for ssim_fn in self.ssim_fns:
            out1 = ssim_fn(img1, imgf)
            out2 = ssim_fn(img2, imgf)
            gamma = out1['sigma'] / (out1['sigma'] +
                                     out2['sigma']).clamp_(min=eps)
            ssim += (gamma * out1['ssim']).mean() + (
                (1.0 - gamma) * out2['ssim']).mean()

        return ssim / len(self.win_sizes)


class SSIMLoss(nn.Module):
    def __init__(self,
                 mode='ssim',
                 data_range=1.0,
                 use_padding=False,
                 weight=1.0):
        super(SSIMLoss, self).__init__()
        self.mode = mode
        self.data_range = data_range
        self.use_padding = use_padding
        self.weight = weight

    def forward(self, img1, img2, imgf):
        if self.mode == 'ssim':
            ssim_fn = SSIM(11, self.data_range, self.use_padding)
            ssim1 = ssim_fn(img1, imgf)['ssim'].mean()
            ssim2 = ssim_fn(img2, imgf)['ssim'].mean()
            loss = (ssim1 + ssim2) * 0.5

        elif self.mode == 'w-ssim':
            ssim_fn = SSIM(11, self.data_range, self.use_padding)
            out1 = ssim_fn(img1, imgf)
            out2 = ssim_fn(img2, imgf)
            gamma = out1['sigma'] / (out1['sigma'] +
                                     out2['sigma']).clamp_(min=eps)
            loss = (gamma * out1['ssim']).mean() + (
                (1.0 - gamma) * out2['ssim']).mean()

        elif self.mode == 'ms-ssim':
            msssim_fn = MS_SSIM(11, self.data_range, self.use_padding)
            msssim1 = msssim_fn(img1, imgf).mean()
            msssim2 = msssim_fn(img2, imgf).mean()
            loss = (msssim1 + msssim2) * 0.5

        elif self.mode == 'msw-ssim':
            mswssim_fn = MSW_SSIM((11, 9, 7, 5, 3), self.data_range,
                                  self.use_padding)
            loss = mswssim_fn(img1, img2, imgf)

        else:
            raise ValueError(
                "only supported ['ssim', 'w-ssim', 'ms-ssim', 'msw-ssim'] mode"
            )

        return self.weight * (1.0 - loss)


class PixelLoss(nn.Module):
    def __init__(self, mode='l1', weight=1.0):
        super(PixelLoss, self).__init__()
        self.mode = mode
        self.weight = weight
        self.loss_fn = NormLoss(mode, weight)

    def forward(self, img1, img2, imgf, mode='avg'):
        if mode == 'avg':
            loss1 = imgf - img1
            loss2 = imgf - img2

            return (self.loss_fn(loss1) + self.loss_fn(loss2)) * 0.5

        elif mode == 'max':
            loss = imgf - torch.max(img1, img2)

            return self.loss_fn(loss)


class GradLoss(nn.Module):
    def __init__(self, mode='l1', weight=1.0):
        super(GradLoss, self).__init__()
        self.mode = mode
        self.weight = weight
        self.loss_fn = NormLoss(mode, weight)

        x_sobel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2],
                                     [-1, 0, 1]]).reshape(1, 1, 3, 3)
        y_sobel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0],
                                     [1, 2, 1]]).reshape(1, 1, 3, 3)

        self.register_buffer('x_sobel', x_sobel)
        self.register_buffer('y_sobel', y_sobel)

    def _sobel_fn(self, img):
        im = F.pad(img.clone(), (1, 1, 1, 1), 'reflect')

        x_grad = F.conv2d(im, self.x_sobel.to(im, non_blocking=True))
        y_grad = F.conv2d(im, self.y_sobel.to(im, non_blocking=True))

        return torch.abs(x_grad) + torch.abs(y_grad)

    def forward(self, img1, img2, imgf, mode='avg'):
        img1 = self._sobel_fn(img1)
        img2 = self._sobel_fn(img2)
        imgf = self._sobel_fn(imgf)

        if mode == 'avg':
            loss1 = imgf - img1
            loss2 = imgf - img2

            return (self.loss_fn(loss1) + self.loss_fn(loss2)) * 0.5

        elif mode == 'max':
            loss = imgf - torch.max(img1, img2)

            return self.loss_fn(loss)


class TVLoss(nn.Module):
    def __init__(self, mode='l1', weight=1.0):
        super(TVLoss, self).__init__()
        self.mode = mode
        self.weight = weight
        self.loss_fn = NormLoss(mode, weight)

    def forward(self, x):
        tv_h = x[..., 1:, :] - x[..., :-1, :]
        tv_w = x[..., :, 1:] - x[..., :, :-1]

        return self.loss_fn(tv_h) + self.loss_fn(tv_w)


class NormLoss(nn.Module):
    def __init__(self, mode='l1', weight=1.0):
        super(NormLoss, self).__init__()
        self.mode = mode
        self.weight = weight

    @staticmethod
    def _l1_loss(x):
        return torch.abs(x).mean()

    @staticmethod
    def _l2_loss(x):
        return torch.pow(x, 2).mean()

    def forward(self, x):
        if self.mode == 'l1':
            loss = self._l1_loss(x)

        elif self.mode == 'l2':
            loss = self._l2_loss(x)

        else:
            raise ValueError("only supported ['l1', 'l2'] mode")

        return self.weight * loss


if __name__ == '__main__':

    torch.manual_seed(0)

    ssim_modes = ['ssim', 'w-ssim', 'ms-ssim', 'msw-ssim']
    norm_modes = ['l1', 'l2']
    weights = [1.0, 0.1, 0.01, 0.001, 0.0]

    ssim_mode, ssim_weight = ssim_modes[0], weights[0]
    pixel_mode, pixel_weight = norm_modes[0], weights[2]
    grad_mode, grad_weight = norm_modes[0], weights[1]

    loss_fn1 = SSIMLoss(ssim_mode, weight=ssim_weight)
    loss_fn2 = PixelLoss(pixel_mode, weight=pixel_weight)
    loss_fn3 = GradLoss(grad_mode, weight=grad_weight)
    loss_fn4 = TVLoss(norm_modes[0], weight=weights[0])

    x1 = torch.rand(2, 1, 256, 256)
    x2 = torch.rand(2, 1, 256, 256)
    y = torch.rand(2, 1, 256, 256)

    loss1 = loss_fn1(x1, x2, y)
    loss2 = loss_fn2(x1, x2, y)
    loss3 = loss_fn3(x1, x2, y)
    loss4 = loss_fn4(y - x1)
    total_loss = loss1 + loss2 + loss3

    print(f'ssim loss: {loss1.item():.4f}')
    print(f'pixel loss: {loss2.item():.4f}')
    print(f'grad loss: {loss3.item():.4f}')
    print(f'tv loss: {loss4.item():.4f}')
    print(f'total loss: {total_loss.item():.4f}')
