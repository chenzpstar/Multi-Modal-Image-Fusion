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

        if set_name is None:
            if set_type == 'train':
                self.data_info = self.train_data_info
            elif set_type == 'valid':
                self.data_info = self.valid_data_info

    def __getitem__(self, index):
        img_path1, img_path2 = self.data_info[index]
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        imgs = (img1, img2)

        imgs = tuple(map(partial(norm, mode=self.norm), imgs))

        if self.transform:
            idx = np.random.choice(2)
            imgs = tuple(map(partial(transform, mode=idx), imgs))

        imgs = tuple(
            map(lambda img: torch.from_numpy(img[None, :].copy()).float(),
                imgs))

        if self.fix_size:
            imgs = torch.stack(imgs, dim=0)

            min_size = min(imgs.shape[-2:])
            if min_size < img_size:
                # imgs = tf.CenterCrop(min_size)(imgs)
                imgs = tf.RandomCrop(min_size)(imgs)
                imgs = tf.Resize(img_size)(imgs)
            else:
                # imgs = tf.CenterCrop(img_size)(imgs)
                imgs = tf.RandomCrop(img_size)(imgs)

            return imgs[0], imgs[1]

        return imgs

    def __len__(self):
        assert len(self.data_info) > 0
        return len(self.data_info)

    def _get_data_info(self):
        img_info1, img_info2 = [], []

        if self.set_name is None:
            img_dir = os.path.join(self.root_dir, 'vi')
        else:
            img_dir = os.path.join(self.root_dir, self.set_name, 'vi')

        for img in sorted(os.listdir(img_dir)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img_path1 = os.path.join(img_dir, img)

                assert self.img_type in ['ir', 'po']
                img_path2 = img_path1.replace('vi', self.img_type)

                if os.path.isfile(img_path2):
                    img_info1.append(img_path1)
                    img_info2.append(img_path2)

        if self.set_name is None and self.set_type in ['train', 'valid']:
            train_img_path1, valid_img_path1, train_img_path2, valid_img_path2 = train_test_split(
                img_info1, img_info2, test_size=0.2, random_state=0)
            self.train_data_info = list(zip(train_img_path1, train_img_path2))
            self.valid_data_info = list(zip(valid_img_path1, valid_img_path2))
        else:
            self.data_info = list(zip(img_info1, img_info2))
            if self.set_type in ['train', 'valid']:
                random.shuffle(self.data_info)


class AEDataset(Dataset):
    def __init__(self,
                 root_dir,
                 img_type='ir',
                 norm=None,
                 transform=False,
                 fix_size=False):
        super(AEDataset, self).__init__()
        self.root_dir = root_dir
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

        img = torch.from_numpy(img[None, :].copy()).float()

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
        img_dir1 = os.path.join(self.root_dir, 'vi')

        assert self.img_type in ['ir', 'po']
        img_dir2 = img_dir1.replace('vi', self.img_type)

        for img in sorted(os.listdir(img_dir1)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img_path = os.path.join(img_dir1, img)
                self.data_info.append(img_path)

        for img in sorted(os.listdir(img_dir2)):
            if img.endswith('.bmp') or img.endswith('.jpg') or img.endswith(
                    '.png'):
                img_path = os.path.join(img_dir2, img)
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
        train_path = os.path.join(BASE_DIR, 'samples')
        train_set = FusionDataset(train_path,
                                  set_name='train',
                                  img_type='po',
                                  norm='min-max',
                                  transform=True,
                                  fix_size=True)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        for img1, img2 in train_loader:
            result = tuple(
                map(denorm, (img1[0], img2[0], (img1[0] + img2[0]) / 2.0)))
            result = np.concatenate(result, axis=1)

            cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('demo', result.shape[1] // 2,
                             result.shape[0] // 2)
            cv2.imshow('demo', result)
            cv2.waitKey()
            cv2.destroyAllWindows()

    if flag == 1:
        train_path = os.path.join(BASE_DIR, 'samples', 'train')
        train_set = AEDataset(train_path,
                              img_type='po',
                              norm='min-max',
                              transform=True,
                              fix_size=True)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        for img in train_loader:
            result = denorm(img[0])

            cv2.namedWindow('demo', cv2.WINDOW_FULLSCREEN)
            cv2.imshow('demo', result)
            cv2.waitKey()
            cv2.destroyAllWindows()
