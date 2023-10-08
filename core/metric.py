# -*- coding: utf-8 -*-
"""
# @file name  : metric.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2023-08-18
# @brief      : 指标函数
# @reference  : https://github.com/Linfeng-Tang/Image-Fusion/blob/main/General%20Evaluation%20Metric/Evaluation
"""

from math import exp, pi

import torch
import torch.nn.functional as F

__all__ = [
    'cmpt_mean', 'cmpt_std', 'cmpt_ag', 'cmpt_sf', 'cmpt_mse', 'cmpt_psnr',
    'cmpt_cc', 'cmpt_scd', 'cmpt_entropy', 'cmpt_cross_ent', 'cmpt_mul_info',
    'cmpt_Qabf', 'cmpt_Nabf', 'cmpt_Labf', 'cmpt_ssim', 'cmpt_msssim',
    'cmpt_viff'
]


# 1.均值 mean
def cmpt_mean(img):
    return img.mean()


# 2.标准差 sd
def cmpt_std(img):
    im = img.clone()
    im -= cmpt_mean(im)

    return im.pow(2).mean().pow(0.5)


# 3.平均梯度 ag
def cmpt_ag(img):
    im = img.clone()

    x_grad = im[..., :-1, 1:] - im[..., :-1, :-1]
    y_grad = im[..., 1:, :-1] - im[..., :-1, :-1]

    grad = ((x_grad.pow(2) + y_grad.pow(2)) * 0.5).pow(0.5)

    return grad.mean()


# 4.空间频率 sf
def cmpt_sf(img):
    im = img.clone()

    y_grad = im[..., 1:, :] - im[..., :-1, :]
    x_grad = im[..., :, 1:] - im[..., :, :-1]

    r_freq = y_grad.pow(2).mean()
    c_freq = x_grad.pow(2).mean()

    return (r_freq + c_freq).pow(0.5)


# 5.均方误差 mse
def cmpt_mse(img1, img2):
    im1 = img1.clone() / 255.0
    im2 = img2.clone() / 255.0
    err = im1 - im2

    return err.pow(2).mean()


# 6.峰值信噪比 psnr
def cmpt_psnr(mse, L=1.0, root=False):
    if root:
        return 20.0 * torch.log10(L / mse**0.5)

    return 10.0 * torch.log10(L**2 / mse)


# 7.相关系数 cc
def cmpt_cc(img1, img2):
    im1 = img1.clone()
    im2 = img2.clone()

    im1 -= cmpt_mean(im1)
    im2 -= cmpt_mean(im2)

    corr12 = (im1 * im2).sum()
    corr11 = (im1 * im1).sum()
    corr22 = (im2 * im2).sum()

    return corr12 / (corr11 * corr22).pow(0.5)


# 8.差异相关性总和 scd
def cmpt_scd(img1, img2, imgf):
    diff1 = imgf - img1
    diff2 = imgf - img2

    return cmpt_cc(diff1, img2) + cmpt_cc(diff2, img1)


# 9.信息熵 en
def cmpt_prob(img):
    im = img.clone().to(torch.uint8)

    gray = torch.zeros(256)

    for i in range(im.shape[-2]):
        for j in range(im.shape[-1]):
            gray[im[..., i, j].item()] += 1

    return gray / im.numel()


def cmpt_entropy(img):
    prob = cmpt_prob(img)

    idx = torch.where(prob != 0)
    en = -prob[idx] * torch.log2(prob[idx])

    return en.sum()


# 10.联合熵 je
def cmpt_joint_prob(img1, img2):
    im1 = img1.clone().to(torch.uint8)
    im2 = img2.clone().to(torch.uint8)

    gray = torch.zeros((256, 256))

    for i in range(im1.shape[-2]):
        for j in range(im1.shape[-1]):
            gray[im1[..., i, j].item(), im2[..., i, j].item()] += 1

    return gray / im1.numel()


def cmpt_joint_ent(img1, img2):
    prob12 = cmpt_joint_prob(img1, img2)

    idx = torch.where(prob12 != 0)
    je = -prob12[idx] * torch.log2(prob12[idx])

    return je.sum()


# 11.交叉熵 ce
def cmpt_cross_ent(img1, img2):
    prob1 = cmpt_prob(img1)
    prob2 = cmpt_prob(img2)

    idx = torch.where(prob1 * prob2 != 0)
    ce = prob1[idx] * torch.log2(prob1[idx] / prob2[idx])

    return ce.sum()


# 12.互信息 mi
def cmpt_mul_info(img1, img2, normalized=False):
    # prob1 = cmpt_prob(img1)
    # prob2 = cmpt_prob(img2)
    # prob12 = cmpt_joint_prob(img1, img2)

    # idx = torch.where(prob1.reshape(-1, 1) * prob2.reshape(1, -1) * prob12 != 0)
    # mi = prob12[idx] * torch.log2(prob12[idx] / (prob1[idx[0]] * prob2[idx[1]]))

    # return mi.sum()

    en1 = cmpt_entropy(img1)
    en2 = cmpt_entropy(img2)
    en12 = cmpt_joint_ent(img1, img2)

    mi = en1 + en2 - en12

    if normalized:
        return 2.0 * mi / (en1 + en2)

    return mi


# 13.Qabf
def _sobel_fn(img):
    x_sobel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2],
                                 [-1, 0, 1]]).reshape(1, 1, 3, 3)
    y_sobel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0],
                                 [1, 2, 1]]).reshape(1, 1, 3, 3)

    im = F.pad(img.clone(), (1, 1, 1, 1), 'reflect')

    x_grad = F.conv2d(im, x_sobel.to(im, non_blocking=True))
    y_grad = F.conv2d(im, y_sobel.to(im, non_blocking=True))

    grad = (x_grad.pow(2) + y_grad.pow(2)).pow(0.5)
    alpha = torch.atan2(y_grad, x_grad)

    return grad, alpha


def cmpt_Qxy(img1, img2, mode='qabf', full=False):
    g1, a1 = _sobel_fn(img1)
    g2, a2 = _sobel_fn(img2)

    G = torch.min(g1, g2) / torch.max(g1, g2)
    G[G != G] = 0.0  # 若为 nan, 则设为 0
    A = torch.abs(torch.abs(a1 - a2) - pi / 2) * 2 / pi

    if mode == 'qabf':  # from original paper
        Gg, kg, sg = 0.9994, 15, 0.5
        Ga, ka, sa = 0.9879, 22, 0.8
    elif mode == 'nabf':  # from matlab code
        Gg, kg, sg = 0.9999, 19, 0.5
        Ga, ka, sa = 0.9995, 22, 0.5

    Qg = Gg / (1 + torch.exp(-kg * (G - sg)))
    Qa = Ga / (1 + torch.exp(-ka * (A - sa)))

    if full:
        return Qg * Qa, g1, g2

    return Qg * Qa, g1


def cmpt_Qabf(img1, img2, imgf, L=1.5, full=False):
    if full:
        Qaf, ga, gf = cmpt_Qxy(img1, imgf, full=full)
    else:
        Qaf, ga = cmpt_Qxy(img1, imgf)

    Qbf, gb = cmpt_Qxy(img2, imgf)

    wa = ga.pow(L)
    wb = gb.pow(L)

    if full:
        AM = torch.where(gf > torch.max(ga, gb), 1.0, 0.0)
        RR = torch.where(gf <= torch.max(ga, gb), 1.0, 0.0)

        qabf = (Qaf * wa + Qbf * wb).sum() / (wa + wb).sum()
        nabf = (AM * ((1.0 - Qaf) * wa +
                      (1.0 - Qbf) * wb)).sum() / (wa + wb).sum()
        labf = (RR * ((1.0 - Qaf) * wa +
                      (1.0 - Qbf) * wb)).sum() / (wa + wb).sum()

        return qabf, nabf, labf

    return (Qaf * wa + Qbf * wb).sum() / (wa + wb).sum()


# 14.Nabf
def cmpt_Nabf(img1, img2, imgf, L=1.5, modified=True):
    Qaf, ga, gf = cmpt_Qxy(img1, imgf, mode='qabf', full=True)
    Qbf, gb = cmpt_Qxy(img2, imgf, mode='qabf')

    wa = ga.pow(L)
    wb = gb.pow(L)

    AM = torch.where(gf > torch.max(ga, gb), 1.0, 0.0)

    if modified:
        return (AM * ((1.0 - Qaf) * wa +
                      (1.0 - Qbf) * wb)).sum() / (wa + wb).sum()

    return (AM * ((2.0 - Qaf - Qbf) * (wa + wb))).sum() / (wa + wb).sum()


# 15.Labf
def cmpt_Labf(img1, img2, imgf, L=1.5):
    Qaf, ga, gf = cmpt_Qxy(img1, imgf, mode='qabf', full=True)
    Qbf, gb = cmpt_Qxy(img2, imgf, mode='qabf')

    wa = ga.pow(L)
    wb = gb.pow(L)

    RR = torch.where(gf <= torch.max(ga, gb), 1.0, 0.0)

    return (RR * ((1.0 - Qaf) * wa + (1.0 - Qbf) * wb)).sum() / (wa + wb).sum()


# 16.ssim
def _gaussian_kernel(win_size=11, sigma=1.5):
    gauss = torch.FloatTensor([
        exp(-(x - win_size // 2)**2 / (2.0 * sigma**2))
        for x in range(win_size)
    ])

    return gauss / gauss.sum()


def create_window(win_size=11, sigma=1.5):
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
              data_range=255.0,
              use_padding=False,
              size_average=True,
              full=False):
    h, w = img1.shape[-2:]
    min_size = min(win_size, h, w)
    window = create_window(min_size)
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
    L = data_range
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    m1 = 2.0 * mu1_mu2 + C1
    m2 = mu1_sq + mu2_sq + C1
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    cs = v1 / v2
    ssim = (m1 * v1) / (m2 * v2)

    if size_average:
        cs = cs.mean()
        ssim = ssim.mean()

    if full:
        return ssim, cs

    return ssim


# 17.msssim
def cmpt_msssim(img1, img2, win_size=11, data_range=255.0, use_padding=False):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights.to(img1, non_blocking=True)

    im1 = img1.clone()
    im2 = img2.clone()

    values = []
    levels = len(weights)

    for i in range(levels):
        ssim, cs = cmpt_ssim(im1,
                             im2,
                             win_size,
                             data_range,
                             use_padding,
                             full=True)

        if i < levels - 1:
            values.append(cs)

            h, w = im1.shape[-2:]
            pad_h, pad_w = h % 2, w % 2
            im1 = F.pad(im1, (0, pad_w, 0, pad_h), 'reflect')
            im2 = F.pad(im2, (0, pad_w, 0, pad_h), 'reflect')

            im1 = F.avg_pool2d(im1, 2, 2)
            im2 = F.avg_pool2d(im2, 2, 2)
        else:
            values.append(ssim)

    values = torch.stack(values, dim=0).clamp(min=1e-7)
    msssim = torch.prod(values**weights, dim=0)

    return msssim


# 18.viff
def cmpt_vif(img1, img2, use_padding=False):
    eps = 1e-10
    sn_sq = 0.005 * 255 * 255
    VID, VIND, G = [], [], []

    im1 = img1.clone()
    im2 = img2.clone()

    for scale in range(1, 5):
        win_size = 2**(4 - scale + 1) + 1
        window = create_window(win_size, win_size / 5)
        window = window.to(img1, non_blocking=True)

        if scale > 1:
            im1 = _gaussian_fn(im1, window, use_padding)
            im2 = _gaussian_fn(im2, window, use_padding)
            im1 = im1[..., ::2, ::2]
            im2 = im2[..., ::2, ::2]

        mu1 = _gaussian_fn(im1, window, use_padding)
        mu2 = _gaussian_fn(im2, window, use_padding)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _gaussian_fn(im1 * im1, window, use_padding) - mu1_sq
        sigma2_sq = _gaussian_fn(im2 * im2, window, use_padding) - mu2_sq
        sigma12 = _gaussian_fn(im1 * im2, window, use_padding) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0.0
        sigma2_sq[sigma2_sq < 0] = 0.0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0.0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0.0

        g[sigma2_sq < eps] = 0.0
        sv_sq[sigma2_sq < eps] = 0.0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0.0

        sv_sq[sv_sq < eps] = eps

        VID.append(torch.log2(1 + g.pow(2) * sigma1_sq / (sv_sq + sn_sq)))
        VIND.append(torch.log2(1 + sigma1_sq / sn_sq))
        G.append(g)

    return VID, VIND, G


def cmpt_viff(img1, img2, imgf, simple=True):
    N1, D1, G1 = cmpt_vif(img1, imgf)
    N2, D2, G2 = cmpt_vif(img2, imgf)

    if simple:
        num1, num2 = 0.0, 0.0
        den1, den2 = 0.0, 0.0

        for i in range(4):
            num1 += N1[i].sum()
            num2 += N2[i].sum()
            den1 += D1[i].sum()
            den2 += D2[i].sum()

        return num1 / den1 + num2 / den2

    else:
        p = torch.FloatTensor([1.0, 0.0, 0.15, 1.0]) / 2.15
        viff = torch.zeros(4)

        for i in range(4):
            num1, num2 = N1[i], N2[i]
            den1, den2 = D1[i], D2[i]
            g1, g2 = G1[i], G2[i]

            num = torch.where(g1 < g2, num1, num2)
            den = torch.where(g1 < g2, den1, den2)

            viff[i] = num.sum() / den.sum()

        return (p * viff).sum()


if __name__ == '__main__':

    import time

    torch.manual_seed(0)

    x1 = torch.rand(1, 1, 256, 256) * 255.0
    x2 = torch.rand(1, 1, 256, 256) * 255.0
    y = torch.rand(1, 1, 256, 256) * 255.0

    start = time.time()

    mean = cmpt_mean(y)
    sd = cmpt_std(y)
    ag = cmpt_ag(y)
    sf = cmpt_sf(y)

    mse = (cmpt_mse(x1, y) + cmpt_mse(x2, y)) * 0.5
    psnr = cmpt_psnr(mse)

    cc = (cmpt_cc(x1, y) + cmpt_cc(x2, y)) * 0.5
    scd = cmpt_scd(x1, x2, y)

    en = cmpt_entropy(y)
    ce = cmpt_cross_ent(x1, y) + cmpt_cross_ent(x2, y)
    mi = cmpt_mul_info(x1, y, normalized=True) + cmpt_mul_info(
        x2, y, normalized=True)

    qabf = cmpt_Qabf(x1, x2, y, L=1.5)
    nabf = cmpt_Nabf(x1, x2, y, L=1.5, modified=True)
    labf = cmpt_Labf(x1, x2, y, L=1.5)

    ssim = (cmpt_ssim(x1, y) + cmpt_ssim(x2, y)) * 0.5
    msssim = (cmpt_msssim(x1, y) + cmpt_msssim(x2, y)) * 0.5

    viff = cmpt_viff(x1, x2, y, simple=False)

    end = time.time()

    print('mean: {:.4f}'.format(mean))
    print('sd: {:.4f}'.format(sd))
    print('ag: {:.4f}'.format(ag))
    print('sf: {:.4f}'.format(sf))
    print('mse: {:.4f}'.format(mse))
    print('psnr: {:.4f}'.format(psnr))
    print('cc: {:.4f}'.format(cc))
    print('scd: {:.4f}'.format(scd))
    print('en: {:.4f}'.format(en))
    print('ce: {:.4f}'.format(ce))
    print('mi: {:.4f}'.format(mi))
    print('qabf: {:.4f}'.format(qabf))
    print('nabf: {:.4f}'.format(nabf))
    print('labf: {:.4f}'.format(labf))
    print('ssim: {:.4f}'.format(ssim))
    print('msssim: {:.4f}'.format(msssim))
    print('viff: {:.4f}'.format(viff))
    print('-' * 16)
    print('time: {:.3f}s'.format(end - start))
