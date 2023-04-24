# -*- coding: utf-8 -*-
"""
# @file name  : eval.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-01
# @brief      : 模型评估
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import argparse
import time

import cv2
import torch
from common import AverageMeter, save_result
from core.block import *
from core.loss import SSIM
from core.model import *
from data.dataset import FusionDataset as Dataset
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--use_gpu',
                        default=False,
                        type=bool,
                        help='enable to eval on gpu')
    parser.add_argument('--data',
                        default='tno',
                        type=str,
                        help='dataset folder name')
    parser.add_argument('--ckpt',
                        default='2023-02-26_23-15',
                        type=str,
                        help='checkpoint folder name')

    return parser.parse_args()


def eval_model(model, data_loader, eval_fn, save_dir=None):
    timer = AverageMeter()
    ssim = AverageMeter()

    for iter, (img1, img2) in enumerate(data_loader):
        if iter > 0:
            start_time = time.time()

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(img1, img2)

        if iter > 0:
            timer.update(time.time() - start_time)

        ssim1 = eval_fn(pred, img1)['ssim'].mean()
        ssim2 = eval_fn(pred, img2)['ssim'].mean()
        avg_ssim = (ssim1 + ssim2) / 2.0
        ssim.update(avg_ssim.item())

        print('iter: {:0>2}, eval ssim: {:.3%}, infer time: {:.3f}ms'.format(
            iter + 1, ssim.val, timer.val * 1000))

        if save_dir is not None:
            # result = save_result(pred[0], img1[0], img2[0])
            result = save_result(pred[0])
            file_name = '{:0>2}.png'.format(iter + 1)
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    return ssim.avg, timer.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()

    args = get_args()
    device = torch.device('cuda:0' if args.use_gpu else 'cpu')

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, '..', 'runs', args.ckpt)
    ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')
    assert os.path.isfile(ckpt_path)

    eval_save_dir = os.path.join(ckpt_dir, 'eval')
    if not os.path.isdir(eval_save_dir):
        os.makedirs(eval_save_dir)

    # 1. data
    # eval_set = Dataset(data_dir, set_name='valid')
    # eval_set = Dataset(data_dir, set_type='valid')
    eval_set = Dataset(data_dir, set_type='test')
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. model
    # models = [
    #     PFNetv1, PFNetv2, DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse,
    #     NestFuse, RFNNest, UNFusion, Res2Fusion, MAFusion, IFCNN, DIFNet, PMGI
    # ]
    #
    # model = models[7]
    # print('model: {}'.format(model.__name__))
    #
    # model = model().to(device, non_blocking=True)

    encoders = [
        ConvBlock, SepConvBlock, MixConvBlock, Res2ConvBlock, ConvFormerBlock,
        MixFormerBlock, Res2FormerBlock
    ]
    decoders = [NestDecoder, LSDecoder, FSDecoder]

    encoder, decoder = encoders[0], decoders[0]
    print('encoder: {}, decoder: {}'.format(encoder.__name__,
                                            decoder.__name__))

    model = MyFusion(encoder, decoder).to(device, non_blocking=True)

    params = sum([param.nelement() for param in model.parameters()])
    print('params: {:.3f}M'.format(params / 1e6))

    model.load_state_dict(torch.load(ckpt_path, map_location=device),
                          strict=False)
    model.eval()

    eval_fn = SSIM(no_luminance=False)

    # 3. eval
    eval_ssim, avg_time = eval_model(model, eval_loader, eval_fn,
                                     eval_save_dir)
    print('eval ssim: {:.3%}, average time: {:.3f}ms'.format(
        eval_ssim, avg_time * 1000))
