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

import time

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
from core.block import *
from core.loss import GradLoss, PixelLoss, SSIMLoss
from core.model import *
from data.dataset import FusionDataset as Dataset
from data.patches import FusionPatches as Patches


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

            imgf = model(img1, img2)
            loss1 = loss_fn1(img1, img2, imgf)
            # loss2 = loss_fn2(img1, img2, imgf, mode='avg')
            # loss3 = loss_fn3(img1, img2, imgf, mode='avg')
            loss2 = loss_fn2(img1, img2, imgf, mode='max')
            loss3 = loss_fn3(img1, img2, imgf, mode='max')
            total_loss = loss1 + loss2 + loss3

            total_loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            if args.warmup and epoch < 1:
                warmup_scheduler.step()
            # else:
            #     iter_scheduler.step()

        elif mode == 'valid':
            with torch.no_grad():
                imgf = model(img1, img2)
                loss1 = loss_fn1(img1, img2, imgf)
                # loss2 = loss_fn2(img1, img2, imgf, mode='avg')
                # loss3 = loss_fn3(img1, img2, imgf, mode='avg')
                loss2 = loss_fn2(img1, img2, imgf, mode='max')
                loss3 = loss_fn3(img1, img2, imgf, mode='max')
                total_loss = loss1 + loss2 + loss3

        if is_distributed:
            total_loss = reduce_value(total_loss, args.local_world_size)
            loss1 = reduce_value(loss1, args.local_world_size)
            loss2 = reduce_value(loss2, args.local_world_size)
            loss3 = reduce_value(loss3, args.local_world_size)

        loss.update(total_loss.item(), len(imgf))

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
                    f'epoch: {epoch_idx:0>2}, iter: {iter_idx:0>3}, {mode} loss: {loss.avg:.4f}'
                )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    cost_time = time.time() - start_time

    if local_rank == 0:
        logger.info(f'cost time: {cost_time:.3f}s\n')

        if save_dir is not None:
            result = save_result(imgf[0], img1[0], img2[0])
            # result = save_result(imgf[0])
            file_name = f'{epoch_idx:0>2}.png'
            cv2.imwrite(os.path.join(save_dir, file_name), result)

    if is_distributed:
        dist.barrier()

    return loss.avg


if __name__ == '__main__':
    # 0. config
    torch.cuda.empty_cache()
    setup_seed(seed=0, deterministic=True)

    ckpt_dir, logger = make_logger(BASE_DIR)

    args = get_train_args()

    lr = args.lr
    batch_size = args.bs
    num_epochs = args.epoch
    betas = (0.9, 0.999)
    milestones = (round(args.epoch * 2 / 3), round(args.epoch * 8 / 9))

    is_distributed = args.local_world_size > 1
    logger.info(
        f'rank: {args.local_rank}, wolrd size: {args.local_world_size}')

    if is_distributed:
        setup_dist(args.local_rank, args.local_world_size)
        local_rank = dist.get_rank()
        device = torch.device('cuda', local_rank)
    else:
        local_rank = args.local_rank
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'training on device: {device}')

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
    if args.data in ['tno']:
        set_name = None
    elif args.data in ['roadscene', 'msrs', 'polar']:
        set_name = 'train'

    if args.use_patches:
        train_set = Patches(data_dir,
                            set_name=set_name,
                            set_type='train',
                            transform=True)
        valid_set = Patches(data_dir, set_name=set_name, set_type='valid')
    else:
        train_set = Dataset(data_dir,
                            set_name=set_name,
                            set_type='train',
                            transform=True,
                            fix_size=True)
        valid_set = Dataset(data_dir,
                            set_name=set_name,
                            set_type='valid',
                            fix_size=True)

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
    classic_model = True
    # classic_model = False

    if classic_model:
        models = [
            DeepFuse, DenseFuse, VIFNet, DBNet, SEDRFuse, NestFuse, RFNNest,
            UNFusion, Res2Fusion, MAFusion, IFCNN, DIFNet, PMGI, PFNetv1,
            PFNetv2
        ]

        model = models[0]
        if local_rank == 0:
            logger.info(f'model: {model.__name__}')

        model = model().to(device, non_blocking=True)
    else:
        encoders = [
            SepConvBlock, MixConvBlock, Res2ConvBlock, ConvFormerBlock,
            MixFormerBlock, Res2FormerBlock, TransformerBlock
        ]
        decoders = [Decoder, LSDecoder, NestDecoder, FSDecoder]
        fusion_methods = ['elem', 'attn', 'concat', 'rfn']
        fusion_modes = ['sum', 'mean', 'max', 'sa', 'ca', 'sca', 'wavg', None]
        down_modes = ['maxpool', 'stride']
        up_modes = ['nearest', 'bilinear']

        # encoder = [encoders[0], encoders[0], encoders[0], encoders[0]]
        encoder = encoders[0]
        decoder = decoders[0]
        fusion_method, fusion_mode = fusion_methods[0], fusion_modes[0]
        down_mode, up_mode = down_modes[0], up_modes[0]

        if local_rank == 0:
            if not isinstance(encoder, list):
                logger.info(f'encoder: {encoder.__name__}')
            else:
                [
                    logger.info(f'encoder{i + 1}: {e.__name__}')
                    for i, e in enumerate(encoder)
                ]
            logger.info(f'decoder: {decoder.__name__}')
            logger.info(
                f'fusion method: {fusion_method}, fusion mode: {fusion_mode}')
            logger.info(f'down mode: {down_mode}, up mode: {up_mode}')

        model = MyFusion(encoder,
                         decoder,
                         bias=False,
                         norm=None,
                         act=nn.ReLU6,
                         fusion_method=fusion_method,
                         fusion_mode=fusion_mode,
                         down_mode=down_mode,
                         up_mode=up_mode,
                         share_weight_levels=4).to(device, non_blocking=True)

    if local_rank == 0:
        params = sum([param.numel() for param in model.parameters()])
        logger.info(f'params: {params / 1e6:.3f}M')

    if is_distributed:
        tmp_path = os.path.join(ckpt_dir, 'init_weights.pth')

        if local_rank == 0:
            torch.save(model.state_dict(), tmp_path)

        dist.barrier()
        model.load_state_dict(
            torch.load(tmp_path, map_location={'cuda:0':
                                               f'cuda:{local_rank}'}))

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    model.train()

    # 3. optimize
    ssim_modes = ['ssim', 'w-ssim', 'ms-ssim', 'msw-ssim']
    norm_modes = ['l1', 'l2']
    weights = [1.0, 0.1, 0.01, 0.001, 0.0]

    ssim_mode, ssim_weight = ssim_modes[0], weights[0]
    pixel_mode, pixel_weight = norm_modes[0], weights[2]
    grad_mode, grad_weight = norm_modes[0], weights[1]

    if local_rank == 0:
        logger.info(f'ssim mode: {ssim_mode}, weight: {ssim_weight}')
        logger.info(f'pixel mode: {pixel_mode}, weight: {pixel_weight}')
        logger.info(f'grad mode: {grad_mode}, weight: {grad_weight}')

    loss_fn1 = SSIMLoss(ssim_mode, weight=ssim_weight)
    loss_fn2 = PixelLoss(pixel_mode, weight=pixel_weight)
    loss_fn3 = GradLoss(grad_mode, weight=grad_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    epoch_scheduler = MultiStepLR(optimizer, milestones, 0.1)
    # iter_scheduler = ExponentialLR(optimizer, 0.99)
    warmup_scheduler = WarmupLR(optimizer, 0.001,
                                len(train_loader)) if args.warmup else None

    # 4. loop
    best_epoch, best_loss = 0, 0.0

    for epoch in range(num_epochs):
        epoch_idx = epoch + 1

        if local_rank == 0:
            logger.info(
                f'Epoch: [{epoch_idx:0>2}/{num_epochs:0>2}], lr: {optimizer.param_groups[0]["lr"]:.2e}'
            )
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
        epoch_scheduler.step()

        if local_rank == 0:
            writer.add_scalar('train_loss_epoch', train_loss, epoch)
            writer.add_scalar('valid_loss_epoch', valid_loss, epoch)

            logger.info(
                f'epoch: {epoch_idx:0>2}, train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}\n'
            )

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
            f'training model done, best loss: {best_loss:.4f} in epoch: {best_epoch}'
        )

    if is_distributed:
        dist.destroy_process_group()
