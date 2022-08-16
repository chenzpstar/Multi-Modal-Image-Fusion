# -*- coding: utf-8 -*-
"""
# @file name  : common.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-08
# @brief      : 通用函数
"""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler

from data.transform import denorm


def save_result(pred, img1=None, img2=None):
    if (img1 is not None) and (img2 is not None):
        result = tuple(map(denorm, (img1, img2, (img1 + img2) / 2.0, pred)))
        result = np.concatenate(result, axis=1)
    else:
        result = denorm(pred)

    return result


def setup_seed(seed=0, benchmark=False, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


def setup_dist(rank=0, world_size=1):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group('nccl')


def reduce_value(value, world_size=1, average=True):
    if world_size == 1:
        return value

    with torch.no_grad():
        dist.all_reduce(value)

        if average:
            value /= world_size

    return value


class WarmupLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_factor=0.001,
                 warmup_iters=1000,
                 warmup_method="linear",
                 last_epoch=-1,
                 verbose=False):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters,
            self.warmup_factor)
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    @staticmethod
    def _get_warmup_factor_at_iter(method, iter, warmup_iters, warmup_factor):
        if iter >= warmup_iters:
            return 1.0

        if method == 'constant':
            return warmup_factor
        elif method == 'linear':
            alpha = iter / warmup_iters
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            raise ValueError("only supported ['constant', 'linear'] method")
