# -*- coding: utf-8 -*-
"""
# @file name  : dataset_patches.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-06
# @brief      : 图像块读取类
"""

import os
from functools import partial

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PolarDataset(Dataset):
    def __init__(self, root_path, norm=None, transform=None):
        super(PolarDataset, self).__init__()
        self.root_path = root_path
        self.norm = norm
        self.transform = transform
        self.data_info = None
        self._get_data_info()

    def __getitem__(self, index):
        patches = self.data_info[index].astype(np.float32)
        imgs = (patches[0], patches[1])

        imgs = tuple(map(lambda img: img / 255.0, imgs))

        if self.norm is not None:
            imgs = tuple(map(partial(self.norm, mode='max-min'), imgs))

        if self.transform is not None:
            idx = np.random.choice(8)
            imgs = tuple(map(partial(self.transform, mode=idx), imgs))

        imgs = tuple(
            map(lambda img: torch.from_numpy(img[None, :].copy()).float(),
                imgs))

        return imgs

    def __len__(self):
        assert len(self.data_info) > 0
        return len(self.data_info)

    def _get_data_info(self):
        patch_path = os.path.join(self.root_path, 'patches', 'patch_224.npy')
        self.data_info = np.load(patch_path)
        np.random.shuffle(self.data_info)


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    from torch.utils.data import DataLoader

    from transform import denorm, norm, transform

    train_path = os.path.join(BASE_DIR, 'samples', 'train')
    train_dataset = PolarDataset(train_path, norm=norm, transform=transform)
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

        cv2.namedWindow('demo', cv2.WINDOW_FULLSCREEN)
        cv2.imshow('demo', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
