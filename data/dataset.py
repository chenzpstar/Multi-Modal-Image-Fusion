# -*- coding: utf-8 -*-
"""
# @file name  : dataset.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-07-26
# @brief      : 数据集读取类
"""

import os
import random
from functools import partial

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .transform import norm, transform
except:
    from transform import norm, transform


class PolarDataset(Dataset):
    def __init__(self, root_path, mode='dolp', norm=None, transform=False):
        super(PolarDataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.norm = norm
        self.transform = transform
        self.data_info = []
        self._get_data_info()

    def __getitem__(self, index):
        vis_path, po_path = self.data_info[index]
        vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
        po_img = cv2.imread(po_path, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'dolp':
            vis = vis_img.astype(np.float32)
            po = po_img.astype(np.float32)
            imgs = (vis, po)

        elif self.mode == 'po':
            h, w = po_img.shape
            half_h, half_w = h // 2, w // 2

            vis = cv2.resize(vis_img, (half_w, half_h))

            po_90 = po_img[:half_h, :half_w].astype(np.float32)
            po_45 = po_img[:half_h, half_w:].astype(np.float32)
            po_135 = po_img[half_h:, :half_w].astype(np.float32)
            po_0 = po_img[half_h:, half_w:].astype(np.float32)

            s0 = (po_0 + po_45 + po_90 + po_135) / 2.0
            s1 = po_0 - po_90
            s2 = po_45 - po_135
            dolp = np.sqrt(s1**2 + s2**2) / np.maximum(s0, 1e-7)

            imgs = (vis, dolp)

        else:
            raise ValueError("only supported ['dolp', 'po'] mode")

        imgs = tuple(map(partial(norm, mode=self.norm), imgs))

        if self.transform:
            idx = np.random.choice(4)
            imgs = tuple(map(partial(transform, mode=idx), imgs))

        imgs = tuple(
            map(lambda img: torch.from_numpy(img[None, :].copy()).float(),
                imgs))

        return imgs

    def __len__(self):
        assert len(self.data_info) > 0
        return len(self.data_info)

    def _get_data_info(self):
        vis_dir = os.path.join(self.root_path, 'vis')

        for img in os.listdir(vis_dir):
            if img.endswith('.bmp') or img.endswith('.jpeg'):
                vis_path = os.path.join(vis_dir, img)

                assert self.mode in ['dolp', 'po']
                po_path = vis_path.replace('vis', self.mode)

                if os.path.isfile(po_path):
                    self.data_info.append((vis_path, po_path))

        random.shuffle(self.data_info)


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    from torch.utils.data import DataLoader

    from transform import denorm

    train_path = os.path.join(BASE_DIR, 'samples', 'train')
    train_dataset = PolarDataset(train_path, norm='min-max', transform=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    for idx, (img1, img2) in enumerate(train_loader):
        result = tuple(
            map(denorm, (img1[0], img2[0], (img1[0] + img2[0]) / 2.0)))
        result = np.concatenate(result, axis=1)

        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('demo', result.shape[1] // 2, result.shape[0] // 2)
        cv2.imshow('demo', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
