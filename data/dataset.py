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
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as tf

try:
    from .transform import norm, transform
except:
    from transform import norm, transform

img_size = 256


class FusionDataset(Dataset):
    def __init__(self,
                 root_dir,
                 set_name=None,
                 set_type='train',
                 img_type='ir',
                 norm=None,
                 transform=False,
                 fix_size=False):
        super(FusionDataset, self).__init__()
        assert set_type in ('train', 'valid', 'test')
        assert img_type in ('ir', 'po')

        self.root_dir = root_dir
        self.set_name = set_name
        self.set_type = set_type
        self.img_type = img_type
        self.norm = norm
        self.transform = transform
        self.fix_size = fix_size
        self.data_info = []
        self.train_data_info = []
        self.valid_data_info = []
        self._get_data_info()

        if set_type == 'train':
            self.data_info = self.train_data_info
        elif set_type == 'valid':
            self.data_info = self.valid_data_info

    def __getitem__(self, index):
        img1_path, img2_path = self.data_info[index]
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        img_pair = (img1, img2)
        img_pair = tuple(map(partial(norm, mode=self.norm), img_pair))

        if self.transform:
            idx = np.random.choice(2)
            img_pair = tuple(map(partial(transform, mode=idx), img_pair))

        img_pair = tuple(
            map(lambda img: torch.from_numpy(img.copy()).float(), img_pair))

        img_pair = torch.stack(img_pair, dim=0)

        if self.fix_size:
            min_size = min(img_pair.shape[-2:])
            if min_size < img_size:
                # img_pair = tf.CenterCrop(min_size)(img_pair)
                img_pair = tf.RandomCrop(min_size)(img_pair)
                img_pair = tf.Resize(img_size)(img_pair)
            else:
                # img_pair = tf.CenterCrop(img_size)(img_pair)
                img_pair = tf.RandomCrop(img_size)(img_pair)

        return torch.chunk(img_pair, 2, dim=0)

    def __len__(self):
        assert len(self.data_info) > 0
        return len(self.data_info)

    def _get_data_info(self):
        img1_info, img2_info = [], []

        if self.set_name is None:
            img_dir = os.path.join(self.root_dir, 'vis')
        else:
            img_dir = os.path.join(self.root_dir, self.set_name, 'vis')

        for img in natsorted(os.listdir(img_dir)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img1_path = os.path.join(img_dir, img)
                img2_path = img1_path.replace('vis', self.img_type)

                if os.path.isfile(img2_path):
                    img1_info.append(img1_path)
                    img2_info.append(img2_path)

        if self.set_type in ('train', 'valid'):
            train_img1_path, valid_img1_path, train_img2_path, valid_img2_path = train_test_split(
                img1_info, img2_info, test_size=0.2, random_state=0)
            self.train_data_info = list(zip(train_img1_path, train_img2_path))
            self.valid_data_info = list(zip(valid_img1_path, valid_img2_path))
        else:
            self.data_info = list(zip(img1_info, img2_info))


class AEDataset(Dataset):
    def __init__(self,
                 root_dir,
                 set_name=None,
                 img_type='ir',
                 norm=None,
                 transform=False,
                 fix_size=False):
        super(AEDataset, self).__init__()
        assert img_type in ('ir', 'po')

        self.root_dir = root_dir
        self.set_name = set_name
        self.img_type = img_type
        self.norm = norm
        self.transform = transform
        self.fix_size = fix_size
        self.data_info = []
        self._get_data_info()

    def __getitem__(self, index):
        img_path = self.data_info[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = norm(img, mode=self.norm)

        if self.transform:
            idx = np.random.choice(2)
            img = transform(img, mode=idx)

        img = torch.from_numpy(img.copy()).float().unsqueeze(0)

        if self.fix_size:
            min_size = min(img.shape[-2:])
            if min_size < img_size:
                # img = tf.CenterCrop(min_size)(img)
                img = tf.RandomCrop(min_size)(img)
                img = tf.Resize(img_size)(img)
            else:
                # img = tf.CenterCrop(img_size)(img)
                img = tf.RandomCrop(img_size)(img)

        return img

    def __len__(self):
        assert len(self.data_info) > 0
        return len(self.data_info)

    def _get_data_info(self):
        if self.set_name is None:
            img1_dir = os.path.join(self.root_dir, 'vis')
        else:
            img1_dir = os.path.join(self.root_dir, self.set_name, 'vis')

        img2_dir = img1_dir.replace('vis', self.img_type)

        for img in natsorted(os.listdir(img1_dir)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img_path = os.path.join(img1_dir, img)
                self.data_info.append(img_path)

        for img in natsorted(os.listdir(img2_dir)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img_path = os.path.join(img2_dir, img)
                self.data_info.append(img_path)

        random.shuffle(self.data_info)


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    from torch.utils.data import DataLoader
    from transform import denorm

    flag = 0
    # flag = 1

    if flag == 0:
        test_path = os.path.join(BASE_DIR, 'samples', 'infrared')
        test_set = FusionDataset(test_path,
                                 set_name='test',
                                 set_type='test',
                                 img_type='ir',
                                 norm='min-max',
                                 transform=True,
                                 fix_size=True)
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for img1, img2 in test_loader:
            result = tuple(
                map(denorm, (img1[0], img2[0], (img1[0] + img2[0]) * 0.5,
                             torch.maximum(img1[0], img2[0]))))
            result = np.concatenate(result, axis=1)

            cv2.namedWindow('demo', cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow('demo', result.shape[1] // 2,
                             result.shape[0] // 2)
            cv2.imshow('demo', result)
            cv2.waitKey()
            cv2.destroyAllWindows()

    if flag == 1:
        train_path = os.path.join(BASE_DIR, 'samples', 'polar')
        train_set = AEDataset(train_path,
                              set_name='test',
                              img_type='po',
                              norm='min-max',
                              transform=True,
                              fix_size=True)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        for img in train_loader:
            result = denorm(img[0])

            cv2.namedWindow('demo', cv2.WINDOW_FULLSCREEN)
            cv2.imshow('demo', result)
            cv2.waitKey()
            cv2.destroyAllWindows()
