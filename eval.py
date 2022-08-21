# -*- coding: utf-8 -*-
"""
# @file name  : eval.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-01
# @brief      : 模型评估
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import argparse
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from common import save_result, setup_seed
from core.loss import SSIM
from core.model import *
from data.dataset import PolarDataset as Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ckpt',
                        default=None,
                        type=str,
                        help='checkpoint folder name')

    return parser.parse_args()


def eval(data_loader, model, eval_fn, device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    model.eval()

    # save_path = os.path.join(ckpt_dir, 'eval')
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)

    eval_ssim = []

    for idx, (vis, dolp) in enumerate(data_loader):
        vis = vis.to(device, non_blocking=True)
        dolp = dolp.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(vis, dolp)
            ssim1 = eval_fn(vis, pred)['ssim'].mean()
            ssim2 = eval_fn(dolp, pred)['ssim'].mean()
            ssim = (ssim1 + ssim2) * 0.5

        eval_ssim.append(ssim.item())
        print('iter: {:0>2}, ssim: {:.3%}'.format(idx + 1, ssim.item()))

        # if idx % 4 == 0:
        #     result = save_result(pred[0], vis[0], dolp[0])
        #     file_name = '{:0>2}.png'.format(idx // 4 + 1)
        #     cv2.imwrite(os.path.join(save_path, file_name), result)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cost_time = time.time() - start_time
    print('\ncost time: {:.0f}m:{:.0f}s'.format(cost_time // 60,
                                                cost_time % 60))

    return np.mean(eval_ssim)


if __name__ == '__main__':
    # 0. config
    args = get_args()

    setup_seed(0)
    torch.cuda.empty_cache()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time_format = args.ckpt if args.ckpt else '2022-08-09_18-50'
    ckpt_dir = os.path.join(BASE_DIR, '..', 'runs', time_format)
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_12.pth')
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')
    ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')

    # 1. data
    eval_path = os.path.join(BASE_DIR, '..', 'data', 'valid')
    eval_dataset = Dataset(eval_path)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
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

    eval_fn = SSIM(val_range=1, no_luminance=False)

    # 3. eval
    eval_ssim = eval(eval_loader, model, eval_fn, device)
    print('eval ssim: {:.3%}'.format(eval_ssim))
