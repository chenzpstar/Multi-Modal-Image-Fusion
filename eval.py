# -*- coding: utf-8 -*-
"""
# @file name  : eval.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-01
# @brief      : 模型评估
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import save_result, setup_seed
from core.loss import SSIM
from core.model import *
from data.dataset import PolarDataset


def eval(eval_loader, model, loss_fn, device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    model.eval()

    eval_ssim = []

    save_path = os.path.join('..', 'results', time_format, 'eval')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for idx, (eval_vis, eval_dolp) in enumerate(eval_loader):
        eval_vis = eval_vis.to(device, non_blocking=True)
        eval_dolp = eval_dolp.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(eval_vis, eval_dolp)
            ssim = (loss_fn(eval_vis, pred)[0] +
                    loss_fn(eval_dolp, pred)[0]) * 0.5

        eval_ssim.append(ssim.item())
        print('iter: {:0>2}, ssim: {:.3%}'.format(idx + 1, ssim.item()))

        if idx % 4 == 0:
            result = save_result(pred[0], eval_vis[0], eval_dolp[0])
            cv2.imwrite(
                os.path.join(save_path, '{:0>2}.png'.format(idx // 4 + 1)),
                result)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cost_time = time.time() - start_time
    print('\ncost time: {:.0f}m:{:.0f}s'.format(cost_time // 60,
                                                cost_time % 60))

    return np.mean(eval_ssim)


if __name__ == '__main__':
    # 0. config
    setup_seed(0)
    torch.cuda.empty_cache()

    batch_size = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time_format = '2022-08-09_18-50'
    ckpt_path = os.path.join('..', 'checkpoints', time_format, 'epoch_12.pth')

    # 1. data
    eval_path = os.path.join('..', 'data', 'valid')
    eval_dataset = PolarDataset(eval_path)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. model
    model = PFNetv1().to(device, non_blocking=True)
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device),
                              strict=False)
    else:
        raise FileNotFoundError(
            'please check your path to checkpoint file: {}'.format(ckpt_path))

    loss_fn = SSIM(val_range=1, size_average=True,
                   full=True).to(device, non_blocking=True)

    # 3. eval
    eval_ssim = eval(
        eval_loader=eval_loader,
        model=model,
        loss_fn=loss_fn,
        device=device,
    )
    print('eval ssim: {:.3%}'.format(eval_ssim))
