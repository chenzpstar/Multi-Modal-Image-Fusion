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

__all__ = [
    'element_fusion', 'weighted_fusion', 'concat_fusion', 'attention_fusion',
    'spatial_fusion', 'channel_fusion', 'spatial_pooling', 'channel_pooling'
]

eps = 1e-7


def element_fusion(tensor1, tensor2, mode='sum'):
    if mode == 'sum':
        return tensor1 + tensor2
    elif mode == 'mean':
        return (tensor1 + tensor2) / 2.0
    elif mode == 'max':
        return torch.max(tensor1, tensor2)
    else:
        raise ValueError("only supported ['sum', 'mean', 'max'] mode")


def weighted_fusion(tensor1, tensor2, w1, w2):
    w = w1 / (w1 + w2).clamp_(min=eps)

    return w * tensor1 + (1.0 - w) * tensor2


def concat_fusion(tensors, dim=1):
    return torch.cat(tensors, dim)


def attention_fusion(tensor1,
                     tensor2,
                     mode='mean',
                     spatial_mode='l1',
                     channel_mode='avg'):
    f_spatial = spatial_fusion(tensor1, tensor2, spatial_mode, softmax=True)
    f_channel = channel_fusion(tensor1, tensor2, channel_mode, softmax=False)

    if mode == 'mean':
        return element_fusion(f_spatial, f_channel, mode)
    elif mode == 'wavg':
        return weighted_fusion(f_spatial, f_channel, f_spatial, f_channel)
    else:
        raise ValueError("only supported ['mean', 'wavg'] mode")


def spatial_fusion(tensor1, tensor2, mode='l1', softmax=True):
    spatial1 = spatial_pooling(tensor1, mode)
    spatial2 = spatial_pooling(tensor2, mode)

    if softmax:
        spatial1 = torch.exp(spatial1)
        spatial2 = torch.exp(spatial2)

    return weighted_fusion(tensor1, tensor2, spatial1, spatial2)


def channel_fusion(tensor1, tensor2, mode='avg', softmax=True):
    channel1 = channel_pooling(tensor1, mode)
    channel2 = channel_pooling(tensor2, mode)

    if softmax:
        channel1 = torch.exp(channel1)
        channel2 = torch.exp(channel2)

    return weighted_fusion(tensor1, tensor2, channel1, channel2)


def spatial_pooling(tensor, mode='l1'):
    if mode == 'sum':
        return tensor.sum(dim=1, keepdim=True)
    elif mode == 'mean':
        return tensor.mean(dim=1, keepdim=True)
    elif mode == 'l1':
        return tensor.norm(p=1, dim=1, keepdim=True)
    elif mode == 'l2':
        return tensor.norm(p=2, dim=1, keepdim=True)
    elif mode == 'linf':
        return tensor.max(dim=1, keepdim=True)[0]
    else:
        raise ValueError(
            "only supported ['sum', 'mean', 'l1', 'l2', 'linf'] mode")


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
    vectors = torch.zeros(1, channel, 1, 1).to(tensor.device,
                                               non_blocking=True)

    for i in range(channel):
        s = torch.svd(tensor[0, i].clamp(min=eps))[1]
        vectors[0, i] = torch.sum(s)

    return vectors
