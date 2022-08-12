# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 模型训练
"""

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from common import *
from core.loss import FusionLoss
from core.model import *
# from data.dataset import PolarDataset
from data.dataset_patches import PolarDataset
from data.transform import transform

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--bs', default=4, type=int, help='batch size')
parser.add_argument('--epoch', default=12, type=int, help='num of epochs')
parser.add_argument('--local_rank',
                    default=0,
                    type=int,
                    help='node rank for distribution')
parser.add_argument("--local_world_size",
                    default=1,
                    type=int,
                    help='num of gpus for distribution')
args = parser.parse_args()


def train(train_loader, model, loss_fn, optimizer, device, epoch):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_time = time.time()

    model.train()

    train_loss = []

    save_path = os.path.join(ckpt_dir, 'train')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for idx, (train_vis, train_dolp) in enumerate(train_loader):
        train_vis = train_vis.to(device, non_blocking=True)
        train_dolp = train_dolp.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(train_vis, train_dolp)
        loss1, loss2 = loss_fn(train_vis, train_dolp, pred)
        total_loss = loss1 + loss2 * 0.1

        total_loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        if is_distributed:
            total_loss = reduce_value(total_loss, args.local_world_size)
            loss1 = reduce_value(loss1, args.local_world_size)
            loss2 = reduce_value(loss2, args.local_world_size)

        train_loss.append(total_loss.item())

        if local_rank == 0:
            writer.add_scalar('train_loss_iter',
                              total_loss.item(),
                              global_step=len(train_loader) * epoch + idx)
            writer.add_scalar('train_loss1_iter',
                              loss1.item(),
                              global_step=len(train_loader) * epoch + idx)
            writer.add_scalar('train_loss2_iter',
                              loss2.item(),
                              global_step=len(train_loader) * epoch + idx)
            writer.add_scalar('lr',
                              optimizer.param_groups[0]['lr'],
                              global_step=len(train_loader) * epoch + idx)

            if (idx + 1) % 10 == 0:
                print(
                    'epoch: {:0>2}, iter: {:0>3}, train loss: {:.4f}, lr: {:.2e}'
                    .format(epoch + 1, idx + 1, total_loss.item(),
                            optimizer.param_groups[0]['lr']))

            if (idx + 1) == len(train_loader):
                for i in range(0, batch_size, batch_size // 4):
                    result = save_result(pred[i], train_vis[i], train_dolp[i])
                    cv2.imwrite(
                        os.path.join(
                            save_path,
                            '{:0>2}_{:0>2}.png'.format(epoch + 1, i + 1)),
                        result)

        if is_distributed:
            dist.barrier()

        if warmup and epoch < 1:
            warmup_scheduler.step()
        # else:
        #     scheduler2.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cost_time = time.time() - start_time

    if local_rank == 0:
        print('\ncost time: {:.0f}m:{:.0f}s'.format(cost_time // 60,
                                                    cost_time % 60))

    return np.mean(train_loss)


def valid(valid_loader, model, loss_fn, device, epoch):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_time = time.time()

    model.eval()

    valid_loss = []

    save_path = os.path.join(ckpt_dir, 'valid')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for idx, (valid_vis, valid_dolp) in enumerate(valid_loader):
        valid_vis = valid_vis.to(device, non_blocking=True)
        valid_dolp = valid_dolp.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(valid_vis, valid_dolp)
            loss1, loss2 = loss_fn(valid_vis, valid_dolp, pred)
            total_loss = loss1 + loss2 * 0.1

        if is_distributed:
            total_loss = reduce_value(total_loss, args.local_world_size)

        valid_loss.append(total_loss.item())

        if local_rank == 0:
            writer.add_scalar('valid_loss_iter',
                              total_loss.item(),
                              global_step=len(valid_loader) * epoch + idx)

            if (idx + 1) % 10 == 0:
                print('epoch: {:0>2}, iter: {:0>3}, valid loss: {:.4f}'.format(
                    epoch + 1, idx + 1, total_loss.item()))

            if (idx + 1) == len(valid_loader):
                for i in range(0, batch_size, batch_size // 4):
                    result = save_result(pred[i], valid_vis[i], valid_dolp[i])
                    cv2.imwrite(
                        os.path.join(
                            save_path,
                            '{:0>2}_{:0>2}.png'.format(epoch + 1, i + 1)),
                        result)

        if is_distributed:
            dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cost_time = time.time() - start_time

    if local_rank == 0:
        print('\ncost time: {:.0f}m:{:.0f}s'.format(cost_time // 60,
                                                    cost_time % 60))

    return np.mean(valid_loss)


if __name__ == '__main__':
    # 0. config
    setup_seed(seed=0, deterministic=False)
    torch.cuda.empty_cache()

    warmup = False
    clip_grad = True

    lr = args.lr if args.lr else 1e-4
    batch_size = args.bs if args.bs else 4
    num_epochs = args.epoch if args.epoch else 12
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

    now = datetime.now()
    time_format = datetime.strftime(now, '%Y-%m-%d_%H-%M')
    ckpt_dir = os.path.join(BASE_DIR, '..', 'runs', time_format)

    if local_rank == 0:
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        writer = SummaryWriter(ckpt_dir)

    # 1. data
    train_path = os.path.join(BASE_DIR, '..', 'data', 'train')
    valid_path = os.path.join(BASE_DIR, '..', 'data', 'valid')

    train_dataset = PolarDataset(train_path, transform=transform)
    valid_dataset = PolarDataset(valid_path)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // args.local_world_size,
        shuffle=False if is_distributed else True,
        sampler=train_sampler if is_distributed else None,
        num_workers=4 * args.local_world_size,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
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

    loss_fn = FusionLoss(mode='mswssim',
                         extra_mode='l1',
                         val_range=1,
                         size_average=False,
                         full=True)

    # 3. optimize
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler1 = MultiStepLR(optimizer, milestones, 0.1)
    # scheduler2 = ExponentialLR(optimizer, 0.99)
    warmup_scheduler = WarmupLR(optimizer, 0.001,
                                len(train_loader)) if warmup else None

    # 4. loop
    best_epoch, best_loss = 0, 10.0

    for epoch in range(num_epochs):
        epoch_idx = epoch + 1

        if local_rank == 0:
            print('Epoch: [{:0>2}/{:0>2}]'.format(epoch_idx, num_epochs))
            print('-' * 16)

        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        if local_rank == 0:
            writer.add_scalar('train_loss_epoch',
                              train_loss,
                              global_step=epoch)
            print('epoch: {:0>2}, train loss: {:.4f}\n'.format(
                epoch_idx, train_loss))

        valid_loss = valid(
            valid_loader=valid_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        if local_rank == 0:
            writer.add_scalar('valid_loss_epoch',
                              valid_loss,
                              global_step=epoch)
            print('epoch: {:0>2}, valid loss: {:.4f}\n'.format(
                epoch_idx, valid_loss))

        scheduler1.step()

        if valid_loss < best_loss:
            best_epoch, best_loss = epoch_idx, valid_loss
            ckpt_path = os.path.join(ckpt_dir, 'epoch_best.pth')

            if is_distributed:
                if local_rank == 0:
                    torch.save(model.module.state_dict(), ckpt_path)

                dist.barrier()
            else:
                torch.save(model.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, 'epoch_last.pth')

    if is_distributed:
        if local_rank == 0:
            torch.save(model.module.state_dict(), ckpt_path)
            os.remove(tmp_path)

        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), ckpt_path)

    if local_rank == 0:
        writer.close()
        print('training model done, best loss: {:.4f} in epoch: {}'.format(
            best_loss, best_epoch))
