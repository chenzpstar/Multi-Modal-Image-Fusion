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
from core.metric import calc_ssim
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
            ssim1 = calc_ssim(img1, imgf, data_range=1.0)
            ssim2 = calc_ssim(img2, imgf, data_range=1.0)
            avg_ssim = (ssim1 + ssim2) * 0.5
            ssim.update(avg_ssim.item())

        print(
            f'iter: {iter_idx:0>2}, ssim: {ssim.val:.4f}, time: {timer.val * 1000:.3f}ms'
        )
        file.write(
            f'\niter: {iter_idx:0>2}, ssim: {ssim.val:.4f}, time: {timer.val * 1000:.3f}ms'
        )

        if save_dir is not None:
            # result = save_result(imgf[0], img1[0], img2[0])
            result = save_result(imgf[0])
            file_name = f'{iter_idx:0>2}.bmp'
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    return ssim.avg, timer.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()

    args = get_test_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:0' if args.gpu else 'cpu')
    # device = torch.device('cpu')

    exp_name = None
    # exp_name = 'exp1'

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    if exp_name is None:
        ckpt_dir = os.path.join(BASE_DIR, '..', 'checkpoints', args.ckpt)
    else:
        ckpt_dir = os.path.join(BASE_DIR, '..', 'checkpoints', exp_name, args.ckpt)
    ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')
    # ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')
    assert os.path.isfile(ckpt_path)

    log_path = os.path.join(ckpt_dir, 'train.log')
    assert os.path.isfile(log_path)

    test_save_dir = os.path.join(ckpt_dir, args.data)
    if not os.path.isdir(test_save_dir):
        os.makedirs(test_save_dir)

    # 1. data
    if args.data in ['tno']:
        set_name = None
    elif args.data in ['roadscene', 'msrs', 'polar']:
        set_name = 'test'

    test_set = Dataset(data_dir, set_name=set_name, set_type='test')
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. model
    # models = [
    #     DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse, NestFuse, RFNNest,
    #     UNFusion, Res2Fusion, MAFusion, IFCNN, DIFNet, PMGI, PFNetv1, PFNetv2
    # ]

    # model = models[0]
    # print(f'model: {model.__name__}')

    # model = model().to(device, non_blocking=True)

    encoders = [
        SepConvBlock, MixConvBlock, Res2ConvBlock, ConvFormerBlock,
        MixFormerBlock, Res2FormerBlock, TransformerBlock
    ]
    decoders = [LSDecoder, NestDecoder, FSDecoder]

    # encoder = [encoders[-2], encoders[-2], encoders[-1], encoders[-1]]
    encoder = encoders[0]
    decoder = decoders[0]

    if not isinstance(encoder, list):
        print(f'encoder: {encoder.__name__}')
    else:
        [print(f'encoder{i + 1}: {e.__name__}') for i, e in enumerate(encoder)]

    print(f'decoder: {decoder.__name__}')

    model = MyFusion(encoder, decoder,
                     share_weight_levels=4).to(device, non_blocking=True)

    params = sum([param.nelement() for param in model.parameters()])
    print(f'params: {params / 1e6:.3f}M')

    model.load_state_dict(torch.load(ckpt_path, map_location=device),
                          strict=False)
    model.eval()

    # 3. test
    with open(log_path, 'a') as file:
        ssim, avg_time = test_model(model, test_loader, test_save_dir)

        print(
            f'ssim: {ssim:.4f}, time: {avg_time * 1000:.3f}ms, fps: {1.0 / avg_time:.3f}'
        )
        file.write(
            f'\nssim: {ssim:.4f}, time: {avg_time * 1000:.3f}ms, fps: {1.0 / avg_time:.3f}'
        )
