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

import cv2
import torch
from torch.utils.data import DataLoader

from common import AverageMeter, save_result
from core.loss import SSIM
from core.model import *
from data.dataset import FusionDataset as Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--data",
                        default="polar",
                        type=str,
                        help="dataset folder name")
    parser.add_argument('--ckpt',
                        default='2022-08-09_18-50',
                        type=str,
                        help='checkpoint folder name')

    return parser.parse_args()


def eval_model(model, data_loader, eval_fn, device, save_dir=None):
    ssim = AverageMeter()

    for iter, (img1, img2) in enumerate(data_loader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(img1, img2)
            ssim1 = eval_fn(img1, pred)['ssim'].mean()
            ssim2 = eval_fn(img2, pred)['ssim'].mean()
            avg_ssim = (ssim1 + ssim2) / 2.0

        ssim.update(avg_ssim.item())
        print('iter: {:0>2}, eval ssim: {:.3%}'.format(iter + 1, ssim.val))

        if save_dir is not None and iter % 4 == 0:
            result = save_result(pred[0], img1[0], img2[0])
            file_name = '{:0>2}.png'.format(iter // 4 + 1)
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    return ssim.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_args()

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, '..', 'runs', args.ckpt)
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_12.pth')
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')
    ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')
    assert os.path.isfile(ckpt_path)

    eval_save_dir = os.path.join(ckpt_dir, 'eval')
    if not os.path.isdir(eval_save_dir):
        os.makedirs(eval_save_dir)

    # 1. data
    eval_set = Dataset(data_dir, 'valid')
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. model
    model = PFNetv1().to(device, non_blocking=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device),
                          strict=False)
    model.eval()

    eval_fn = SSIM(val_range=1, no_luminance=False)

    # 3. eval
    eval_ssim = eval_model(model, eval_loader, eval_fn, device, eval_save_dir)
    print('eval ssim: {:.3%}'.format(eval_ssim))
