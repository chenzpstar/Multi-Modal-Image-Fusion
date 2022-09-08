# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 模型训练
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import argparse
import time
from datetime import datetime

import cv2
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from common import *
from core.loss import PixelLoss, SSIMLoss
from core.model import *
from data.dataset import FusionDataset as Dataset
from data.patches import FusionPatches as Patches


def get_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', default=12, type=int, help='num of epochs')
    parser.add_argument('--use_patches',
                        default=True,
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
    parser.add_argument("--local_world_size",
                        default=1,
                        type=int,
                        help='num of gpus for distribution')
    parser.add_argument("--data",
                        default="polar",
                        type=str,
                        help="dataset folder name")

    return parser.parse_args()


def train_model(model,
                data_loader,
                loss_fn1,
                loss_fn2,
                epoch,
                writer,
                device,
                mode='train',
                optimizer=None,
                save_dir=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_time = time.time()

    loss = AverageMeter()

    epoch_idx = epoch + 1
    num_iters = len(data_loader)

    for iter, (img1, img2) in enumerate(data_loader):
        iter_idx = iter + 1

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)

        if mode == 'train':
            optimizer.zero_grad(set_to_none=True)

            pred = model(img1, img2)
            loss1 = loss_fn1(img1, img2, pred)
            loss2 = loss_fn2((img1 + img2) / 2.0, pred)
            total_loss = loss1 + loss2 * 0.1

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
                loss2 = loss_fn2((img1 + img2) / 2.0, pred)
                total_loss = loss1 + loss2 * 0.1

        if is_distributed:
            total_loss = reduce_value(total_loss, args.local_world_size)
            loss1 = reduce_value(loss1, args.local_world_size)
            loss2 = reduce_value(loss2, args.local_world_size)

        loss.update(total_loss.item(), len(pred))

        if local_rank == 0:
            global_step = num_iters * epoch + iter
            writer.add_scalar(mode + '_loss_iter', total_loss.item(),
                              global_step)
            writer.add_scalar(mode + '_loss1_iter', loss1.item(), global_step)
            writer.add_scalar(mode + '_loss2_iter', loss2.item(), global_step)

            if mode == 'train':
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                  global_step)

            if iter_idx % 10 == 0:
                print(
                    'epoch: {:0>2}, iter: {:0>3}, {} loss: {:.4f}, lr: {:.2e}'.
                    format(epoch_idx, iter_idx, mode, loss.avg,
                           optimizer.param_groups[0]['lr']))

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cost_time = time.time() - start_time

    if local_rank == 0:
        print('\ncost time: {:.0f}m:{:.0f}s'.format(cost_time // 60,
                                                    cost_time % 60))

        if save_dir is not None:
            stride = len(pred) // 4 if len(pred) > 8 else len(pred) // 2

            for i in range(0, len(pred), stride):
                result = save_result(pred[i], img1[i], img2[i])
                file_name = '{:0>2}_{:0>2}.png'.format(epoch_idx, i + 1)
                cv2.imwrite(os.path.join(save_dir, file_name), result)

    if is_distributed:
        dist.barrier()

    return loss.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()
    setup_seed(seed=0, deterministic=False)

    args = get_args()

    lr = args.lr
    batch_size = args.bs
    num_epochs = args.epoch
    betas = (0.9, 0.999)
    milestones = (8, 11)

    is_distributed = args.local_world_size > 1
    print('rank: {}, wolrd size: {}'.format(args.local_rank,
                                            args.local_world_size))

    if is_distributed:
        setup_dist(args.local_rank, args.local_world_size)
        local_rank = dist.get_rank()
        device = torch.device('cuda', local_rank)
    else:
        local_rank = args.local_rank
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('training on device: {}'.format(device))

    data_dir = os.path.join(BASE_DIR, '..', 'datasets', args.data)
    assert os.path.isdir(data_dir)

    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')
    ckpt_dir = os.path.join(BASE_DIR, '..', 'runs', time_str)

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
        train_set = Patches(data_dir, 'train', transform=True)
        valid_set = Patches(data_dir, 'valid')
    else:
        train_set = Dataset(data_dir, 'train', transform=True)
        valid_set = Dataset(data_dir, 'valid')

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
    model = PFNetv1().to(device, non_blocking=True)

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
    loss_fn1 = SSIMLoss(mode='msw-ssim', val_range=1, no_luminance=False)
    loss_fn2 = PixelLoss(mode='l1')

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
            print('Epoch: [{:0>2}/{:0>2}]'.format(epoch_idx, num_epochs))
            print('-' * 16)

        if is_distributed:
            train_sampler.set_epoch(epoch)

        # 1. train
        model.train()
        train_loss = train_model(model, train_loader, loss_fn1, loss_fn2,
                                 epoch, writer, device, 'train', optimizer,
                                 train_save_dir)

        if local_rank == 0:
            writer.add_scalar('train_loss_epoch', train_loss, epoch)
            print('epoch: {:0>2}, train loss: {:.4f}\n'.format(
                epoch_idx, train_loss))

        # 2. valid
        model.eval()
        valid_loss = train_model(model, valid_loader, loss_fn1, loss_fn2,
                                 epoch, writer, device, 'valid',
                                 valid_save_dir)

        if local_rank == 0:
            writer.add_scalar('valid_loss_epoch', valid_loss, epoch)
            print('epoch: {:0>2}, valid loss: {:.4f}\n'.format(
                epoch_idx, valid_loss))

        # 3. update lr
        scheduler1.step()

        if local_rank == 0:
            if valid_loss < best_loss or epoch == 0:
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
        print('training model done, best loss: {:.4f} in epoch: {}'.format(
            best_loss, best_epoch))

    if is_distributed:
        dist.destroy_process_group()
