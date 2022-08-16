# -*- coding: utf-8 -*-
"""
# @file name  : fusion.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-14
# @brief      : 融合函数
# @reference  : https://github.com/hli1221/imagefusion-nestfuse/blob/master/fusion_strategy.py
"""

import torch
import torch.nn.functional as F

eps = 1e-7


def element_fusion(tensor1, tensor2, mode='add'):
    if mode == 'add':
        return tensor1 + tensor2
    elif mode == 'mean':
        return (tensor1 + tensor2) / 2.0
    elif mode == 'max':
        return torch.max(tensor1, tensor2)
    else:
        raise ValueError("only supported ['add', 'mean', 'max'] mode")


def concat_fusion(tensor1, tensor2, dim=1):
    return torch.cat((tensor1, tensor2), dim)


def attention_fusion(tensor1,
                     tensor2,
                     mode='mean',
                     spatial_mode='l1',
                     channel_mode='avg'):
    f_spatial = spatial_fusion(tensor1, tensor2, spatial_mode)
    f_channel = channel_fusion(tensor1, tensor2, channel_mode)

    if mode == 'mean':
        return (f_spatial + f_channel) / 2.0
    elif mode == 'wavg':
        f_w = f_spatial / (f_spatial + f_channel).clamp_(min=eps)
        return f_w * f_spatial + (1.0 - f_w) * f_channel
    else:
        raise ValueError("only supported ['mean', 'wavg'] mode")


def spatial_fusion(tensor1, tensor2, mode='l1', softmax=True):
    channel = tensor1.shape[1]

    spatial1 = spatial_pooling(tensor1, mode)
    spatial2 = spatial_pooling(tensor2, mode)

    if softmax:
        spatial1 = torch.exp(spatial1)
        spatial2 = torch.exp(spatial2)

    spatial_w = spatial1 / (spatial1 + spatial2).clamp_(min=eps)
    spatial_w = spatial_w.repeat(1, channel, 1, 1)

    return spatial_w * tensor1 + (1.0 - spatial_w) * tensor2


def channel_fusion(tensor1, tensor2, mode='avg', softmax=True):
    h, w = tensor1.shape[-2:]

    channel1 = channel_pooling(tensor1, mode)
    channel2 = channel_pooling(tensor2, mode)

    if softmax:
        channel1 = torch.exp(channel1)
        channel2 = torch.exp(channel2)

    channel_w = channel1 / (channel1 + channel2).clamp_(min=eps)
    channel_w = channel_w.repeat(1, 1, h, w)

    return channel_w * tensor1 + (1.0 - channel_w) * tensor2


def spatial_pooling(tensor, mode='l1'):
    if mode == 'sum':
        return tensor.sum(dim=1, keepdim=True)
    elif mode == 'mean':
        return tensor.mean(dim=1, keepdim=True)
    elif mode == 'l1':
        return torch.abs(tensor).sum(dim=1, keepdim=True)
    else:
        raise ValueError("only supported ['sum', 'mean'] mode")


def channel_pooling(tensor, mode='avg'):
    h, w = tensor.shape[-2:]

    if mode == 'avg':
        return F.avg_pool2d(tensor, kernel_size=(h, w))
    elif mode == 'max':
        return F.max_pool2d(tensor, kernel_size=(h, w))
    elif mode == 'nuclear':
        return _nuclear_pooling(tensor)
    else:
        raise ValueError("only supported ['avg', 'max', 'nuclear'] mode")


def _nuclear_pooling(tensor):
    channel = tensor.shape[1]
    vectors = torch.zeros(1, channel, 1, 1).to(tensor.device)

    for i in range(channel):
        s = torch.svd(tensor[0, i].clamp(min=eps))[1]
        vectors[0, i, 0, 0] = torch.sum(s)

    return vectors
