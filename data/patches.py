# -*- coding: utf-8 -*-
"""
# @file name  : patches.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-06
# @brief      : 图像块读取类
"""

import os
import random
from functools import partial

import cv2
import numpy as np
import torch
from natsort import natsorted
from patchify import patchify
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from .transform import norm, transform
except:
    from transform import norm, transform

patch_size = 64
patch_step = 64


class FusionPatches(Dataset):
    def __init__(self,
                 root_dir,
                 set_name=None,
                 set_type='train',
                 img_type='ir',
                 norm=None,
                 transform=False):
        super(FusionPatches, self).__init__()
        assert set_type in ('train', 'valid', 'test')
        assert img_type in ('ir', 'po')

        self.root_dir = root_dir
        self.set_name = set_name
        self.set_type = set_type
        self.img_type = img_type
        self.norm = norm
        self.transform = transform
        self.data_info = []
        self.train_data_info = []
        self.valid_data_info = []
        self._get_data_info()

        if set_type == 'train':
            self.data_info = self.train_data_info
        elif set_type == 'valid':
            self.data_info = self.valid_data_info

        self.patch_pairs = []
        self._gen_patch_pairs()

    def __getitem__(self, index):
        patch_pair = self.patch_pairs[index]
        patch_pair = tuple(map(partial(norm, mode=self.norm), patch_pair))

        if self.transform:
            idx = np.random.choice(8)
            patch_pair = tuple(map(partial(transform, mode=idx), patch_pair))

        patch_pair = tuple(
            map(
                lambda patch: torch.from_numpy(patch.copy()).float().unsqueeze(
                    0), patch_pair))

        return patch_pair

    def __len__(self):
        assert len(self.patch_pairs) > 0
        return len(self.patch_pairs)

    def _get_data_info(self):
        img_info1, img_info2 = [], []

        if self.set_name is None:
            img_dir = os.path.join(self.root_dir, 'vi')
        else:
            img_dir = os.path.join(self.root_dir, self.set_name, 'vi')

        for img in natsorted(os.listdir(img_dir)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img1_path = os.path.join(img_dir, img)
                img2_path = img1_path.replace('vi', self.img_type)

                if os.path.isfile(img2_path):
                    img_info1.append(img1_path)
                    img_info2.append(img2_path)

        if self.set_type in ('train', 'valid'):
            train_img1_path, valid_img1_path, train_img2_path, valid_img2_path = train_test_split(
                img_info1, img_info2, test_size=0.25, random_state=0)
            self.train_data_info = list(zip(train_img1_path, train_img2_path))
            self.valid_data_info = list(zip(valid_img1_path, valid_img2_path))
        else:
            self.data_info = list(zip(img_info1, img_info2))

    def _gen_patch_pairs(self):
        for img1_path, img2_path in self.data_info:
            img1 = cv2.imread(img1_path,
                              cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img2 = cv2.imread(img2_path,
                              cv2.IMREAD_GRAYSCALE).astype(np.float32)

            patches1 = patchify(img1, (patch_size, patch_size),
                                step=patch_step)
            patches2 = patchify(img2, (patch_size, patch_size),
                                step=patch_step)

            patches1 = np.reshape(patches1, (-1, patch_size, patch_size))
            patches2 = np.reshape(patches2, (-1, patch_size, patch_size))

            self.patch_pairs.extend(list(zip(patches1, patches2)))

        random.shuffle(self.patch_pairs)


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    from torch.utils.data import DataLoader
    from transform import denorm

    test_path = os.path.join(BASE_DIR, 'samples', 'msrs')
    test_set = FusionPatches(test_path,
                             set_name='test',
                             set_type='test',
                             img_type='ir',
                             transform=True)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for img1, img2 in test_loader:
        result = tuple(
            map(denorm,
                (img1[0], img2[0],
                 (img1[0] + img2[0]) * 0.5, torch.maximum(img1[0], img2[0]))))
        result = np.concatenate(result, axis=1)

        cv2.namedWindow('demo', cv2.WINDOW_FULLSCREEN)
        cv2.imshow('demo', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
