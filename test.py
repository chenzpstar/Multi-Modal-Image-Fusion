# -*- coding: utf-8 -*-
"""
# @file name  : test.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-01
# @brief      : 模型测试
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import time

import cv2
import torch
from torch.utils.data import DataLoader

from common import AverageMeter, get_test_args, save_result
from core.block import *
from core.metric import cmpt_ssim
from core.model import *
from data.dataset import FusionDataset as Dataset


def test_model(model, data_loader, save_dir=None):
    timer = AverageMeter()
    ssim = AverageMeter()

    for iter, (img1, img2) in enumerate(data_loader):
        iter_idx = iter + 1

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        if iter > 0:
            start_time = time.time()

        with torch.no_grad():
            imgf = model(img1, img2)

        if iter > 0:
            timer.update(time.time() - start_time)

        with torch.no_grad():
            ssim1 = cmpt_ssim(img1, imgf, data_range=1.0)
            ssim2 = cmpt_ssim(img2, imgf, data_range=1.0)
            avg_ssim = (ssim1 + ssim2) * 0.5
            ssim.update(avg_ssim.item())

        print('iter: {:0>2}, ssim: {:.4f}, time: {:.3f}ms'.format(
            iter_idx, ssim.val, timer.val * 1000))

        if save_dir is not None:
            # result = save_result(imgf[0], img1[0], img2[0])
            result = save_result(imgf[0])
            file_name = '{:0>2}.png'.format(iter_idx)
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    return ssim.avg, timer.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()

    args = get_test_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0' if args.gpu else 'cpu')
    # device = torch.device('cpu')

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    ckpt_dir = os.path.join(BASE_DIR, '..', 'checkpoints', args.ckpt)
    ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')
    assert os.path.isfile(ckpt_path)

    test_save_dir = os.path.join(ckpt_dir, 'test')
    if not os.path.isdir(test_save_dir):
        os.makedirs(test_save_dir)

    # 1. data
    # test_set = Dataset(data_dir, set_name='test', set_type='test')
    test_set = Dataset(data_dir, set_type='test')
    test_loader = DataLoader(
        test_set,
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

    model = MyFusion(encoder, decoder,
                     share_weight_levels=4).to(device, non_blocking=True)

    params = sum([param.nelement() for param in model.parameters()])
    print('params: {:.3f}M'.format(params / 1e6))

    model.load_state_dict(torch.load(ckpt_path, map_location=device),
                          strict=False)
    model.eval()

    # 3. test
    ssim, avg_time = test_model(model, test_loader, test_save_dir)
    print('ssim: {:.4f}, fps: {:.3f}'.format(ssim, 1.0 / avg_time))
