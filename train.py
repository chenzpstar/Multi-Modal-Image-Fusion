# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 模型训练
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
import torch.distributed as dist
import torch.nn as nn
from common import *
from core.block import *
from core.loss import GradLoss, PixelLoss, SSIMLoss
from core.model import *
from data.dataset import FusionDataset as Dataset
from data.patches import FusionPatches as Patches
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', default=36, type=int, help='num of epochs')
    parser.add_argument('--use_patches',
                        default=False,
                        type=bool,
                        help='enable to train with patches')
    parser.add_argument('--warmup',
                        default=False,
                        type=bool,
                        help='enable to warm up lr')
    parser.add_argument('--clip_grad',
                        default=True,
                        type=bool,
                        help='enable to clip grad norm')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distribution')
    parser.add_argument('--local_world_size',
                        default=1,
                        type=int,
                        help='num of gpus for distribution')
    parser.add_argument('--data',
                        default='roadscene',
                        type=str,
                        help='dataset folder name')

    return parser.parse_args()


def train_model(model,
                data_loader,
                loss_fn1,
                loss_fn2,
                loss_fn3,
                epoch,
                mode='train',
                save_dir=None):
    loss = AverageMeter()

    epoch_idx = epoch + 1
    num_iters = len(data_loader)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_time = time.time()

    for iter, (img1, img2) in enumerate(data_loader):
        iter_idx = iter + 1

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        if mode == 'train':
            optimizer.zero_grad(set_to_none=True)

            pred = model(img1, img2)
            loss1 = loss_fn1(img1, img2, pred)
            loss2 = loss_fn2(img1, img2, pred, mode='avg')
            loss3 = loss_fn3(img1, img2, pred, mode='avg')
            # loss2 = loss_fn2(img1, img2, pred, mode='max')
            # loss3 = loss_fn3(img1, img2, pred, mode='max')
            total_loss = loss1 + loss2 + loss3

            total_loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            if args.warmup and epoch < 1:
                warmup_scheduler.step()
            # else:
            #     scheduler2.step()

        elif mode == 'valid':
            with torch.no_grad():
                pred = model(img1, img2)
                loss1 = loss_fn1(img1, img2, pred)
                loss2 = loss_fn2(img1, img2, pred, mode='avg')
                loss3 = loss_fn3(img1, img2, pred, mode='avg')
                # loss2 = loss_fn2(img1, img2, pred, mode='max')
                # loss3 = loss_fn3(img1, img2, pred, mode='max')
                total_loss = loss1 + loss2 + loss3

        if is_distributed:
            total_loss = reduce_value(total_loss, args.local_world_size)
            loss1 = reduce_value(loss1, args.local_world_size)
            loss2 = reduce_value(loss2, args.local_world_size)
            loss3 = reduce_value(loss3, args.local_world_size)

        loss.update(total_loss.item(), len(pred))

        if local_rank == 0:
            global_step = num_iters * epoch + iter
            writer.add_scalar(mode + '_loss_iter', total_loss.item(),
                              global_step)
            writer.add_scalar(mode + '_loss1_iter', loss1.item(), global_step)
            writer.add_scalar(mode + '_loss2_iter', loss2.item(), global_step)
            writer.add_scalar(mode + '_loss3_iter', loss3.item(), global_step)

            if mode == 'train':
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                  global_step)

            if iter_idx % 10 == 0:
                logger.info(
                    'epoch: {:0>2}, iter: {:0>3}, {} loss: {:.4f}'.format(
                        epoch_idx, iter_idx, mode, loss.avg))

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cost_time = time.time() - start_time

    if local_rank == 0:
        logger.info('cost time: {:.3f}s\n'.format(cost_time))

        if save_dir is not None:
            result = save_result(pred[0], img1[0], img2[0])
            # result = save_result(pred[0])
            file_name = '{:0>2}.png'.format(epoch_idx)
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    if is_distributed:
        dist.barrier()

    return loss.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()
    setup_seed(seed=0, deterministic=False)

    ckpt_dir, logger = make_logger(BASE_DIR)

    args = get_args()

    lr = args.lr
    batch_size = args.bs
    num_epochs = args.epoch
    betas = (0.9, 0.999)
    milestones = (round(args.epoch * 2 / 3), round(args.epoch * 8 / 9))

    is_distributed = args.local_world_size > 1
    logger.info('rank: {}, wolrd size: {}'.format(args.local_rank,
                                                  args.local_world_size))

    if is_distributed:
        setup_dist(args.local_rank, args.local_world_size)
        local_rank = dist.get_rank()
        device = torch.device('cuda', local_rank)
    else:
        local_rank = args.local_rank
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('training on device: {}'.format(device))

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    if local_rank == 0:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        writer = SummaryWriter(ckpt_dir)

        train_save_dir = os.path.join(ckpt_dir, 'train')
        if not os.path.isdir(train_save_dir):
            os.makedirs(train_save_dir)

        valid_save_dir = os.path.join(ckpt_dir, 'valid')
        if not os.path.isdir(valid_save_dir):
            os.makedirs(valid_save_dir)

    # 1. data
    if args.use_patches:
        train_set = Patches(data_dir, set_name='train', transform=True)
        valid_set = Patches(data_dir, set_name='valid')
    else:
        # train_set = Dataset(data_dir, set_name='train', transform=True)
        # valid_set = Dataset(data_dir, set_name='valid')
        train_set = Dataset(data_dir,
                            set_type='train',
                            transform=True,
                            fix_size=True)
        valid_set = Dataset(data_dir, set_type='valid', fix_size=True)

    if is_distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        valid_sampler = DistributedSampler(valid_set, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size // args.local_world_size,
        shuffle=False if is_distributed else True,
        sampler=train_sampler if is_distributed else None,
        num_workers=4 * args.local_world_size,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size // args.local_world_size,
        shuffle=False,
        sampler=valid_sampler if is_distributed else None,
        num_workers=4 * args.local_world_size,
        pin_memory=True,
    )

    # 2. model
    # models = [
    #     PFNetv1, PFNetv2, DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse,
    #     NestFuse, RFNNest, UNFusion, Res2Fusion, MAFusion, IFCNN, DIFNet, PMGI
    # ]
    #
    # model = models[7]
    # if local_rank == 0:
    #     logger.info('model: {}'.format(model.__name__))
    #
    # model = model().to(device, non_blocking=True)

    encoders = [
        ConvBlock, SepConvBlock, MixConvBlock, Res2ConvBlock, ConvFormerBlock,
        MixFormerBlock, Res2FormerBlock
    ]
    decoders = [NestDecoder, LSDecoder, FSDecoder]

    encoder, decoder = encoders[0], decoders[0]
    if local_rank == 0:
        logger.info('encoder: {}, decoder: {}'.format(encoder.__name__,
                                                      decoder.__name__))

    model = MyFusion(encoder, decoder).to(device, non_blocking=True)

    if local_rank == 0:
        params = sum([param.nelement() for param in model.parameters()])
        logger.info('params: {:.3f}M'.format(params / 1e6))

    if is_distributed:
        tmp_path = os.path.join(ckpt_dir, 'init_weights.pth')

        if local_rank == 0:
            torch.save(model.state_dict(), tmp_path)

        dist.barrier()
        model.load_state_dict(
            torch.load(tmp_path,
                       map_location={'cuda:0': 'cuda:{}'.format(local_rank)}))

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    model.train()

    # 3. optimize
    ssim_modes = ['ssim', 'w-ssim', 'ms-ssim', 'msw-ssim']
    norm_modes = ['l1', 'l2']
    weights = [1.0, 0.1, 0.01, 0.0]

    ssim_mode, ssim_weight = ssim_modes[0], weights[0]
    pixel_mode, pixel_weight = norm_modes[0], weights[2]
    grad_mode, grad_weight = norm_modes[0], weights[1]
    if local_rank == 0:
        logger.info('ssim mode: {}, weight: {}'.format(ssim_mode, ssim_weight))
        logger.info('pixel mode: {}, weight: {}'.format(
            pixel_mode, pixel_weight))
        logger.info('grad mode: {}, weight: {}'.format(grad_mode, grad_weight))

    loss_fn1 = SSIMLoss(ssim_mode, no_luminance=False, weight=ssim_weight)
    loss_fn2 = PixelLoss(pixel_mode, weight=pixel_weight)
    loss_fn3 = GradLoss(grad_mode, weight=grad_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    scheduler1 = MultiStepLR(optimizer, milestones, 0.1)
    # scheduler2 = ExponentialLR(optimizer, 0.99)
    warmup_scheduler = WarmupLR(optimizer, 0.001,
                                len(train_loader)) if args.warmup else None

    # 4. loop
    best_epoch, best_loss = 0, 0.0

    for epoch in range(num_epochs):
        epoch_idx = epoch + 1

        if local_rank == 0:
            logger.info('Epoch: [{:0>2}/{:0>2}], lr: {:.2e}'.format(
                epoch_idx, num_epochs, optimizer.param_groups[0]['lr']))
            logger.info('-' * 16)

        if is_distributed:
            train_sampler.set_epoch(epoch)

        # 1. train
        model.train()
        train_loss = train_model(model, train_loader, loss_fn1, loss_fn2,
                                 loss_fn3, epoch, 'train', train_save_dir)

        # 2. valid
        model.eval()
        valid_loss = train_model(model, valid_loader, loss_fn1, loss_fn2,
                                 loss_fn3, epoch, 'valid', valid_save_dir)

        # 3. update lr
        scheduler1.step()

        if local_rank == 0:
            writer.add_scalar('train_loss_epoch', train_loss, epoch)
            writer.add_scalar('valid_loss_epoch', valid_loss, epoch)

            logger.info(
                'epoch: {:0>2}, train loss: {:.4f}, valid loss: {:.4f}\n'.
                format(epoch_idx, train_loss, valid_loss))

            if epoch < num_epochs // 2:
                continue
            elif valid_loss < best_loss or epoch == num_epochs // 2:
                best_epoch, best_loss = epoch_idx, valid_loss
                ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')

                if is_distributed:
                    torch.save(model.module.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)

        if is_distributed:
            dist.barrier()

    if local_rank == 0:
        ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')

        if is_distributed:
            torch.save(model.module.state_dict(), ckpt_path)
            os.remove(tmp_path)
        else:
            torch.save(model.state_dict(), ckpt_path)

        writer.close()
        logger.info(
            'training model done, best loss: {:.4f} in epoch: {}'.format(
                best_loss, best_epoch))

    if is_distributed:
        dist.destroy_process_group()
