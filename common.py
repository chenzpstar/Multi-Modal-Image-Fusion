# -*- coding: utf-8 -*-
"""
# @file name  : common.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-08
# @brief      : 通用函数
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from data.transform import denorm
from torch.optim.lr_scheduler import _LRScheduler


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
    if world_size > 1:
        with torch.no_grad():
            dist.all_reduce(value)

            if average:
                value /= world_size

    return value


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def is_empty(self):
        return self.count == 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


class Logger(object):
    def __init__(self, log_path):
        log_name = os.path.basename(log_path)
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_name = log_name if log_name else "train.log"
        self.log_path = log_path

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 配置文件 handler
        file_handler = logging.FileHandler(self.log_path, "w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        # 配置屏幕 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(log_formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(root_dir):
    time_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
    log_dir = os.path.join(root_dir, '..', 'runs', time_str)

    # 创建 logger
    log_path = os.path.join(log_dir, "train.log")
    logger = Logger(log_path).init_logger()

    return log_dir, logger
