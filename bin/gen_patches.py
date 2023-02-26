# -*- coding: utf-8 -*-
"""
# @file name  : gen_patches.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-05
# @brief      : 图像块生成
# @reference  : https://github.com/Junchao2018/Polarization-image-fusion/blob/master/GenerateTrainingPatches_Tensorflow.m
"""

import os

import cv2
import numpy as np


def count_patches(data_dir, patch_size, stride):
    count = 0
    img_dir = os.path.join(data_dir, 'vi')

    for img in os.listdir(img_dir):
        img_path1 = os.path.join(img_dir, img)
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

        h, w = img1.shape

        # h_start = (h - patch_size) % stride // 2
        h_start = stride // 2
        h_end = patch_size + h_start

        # w_start = (w - patch_size) % stride // 2
        w_start = stride // 2
        w_end = patch_size + w_start

        for _ in range(h_start, h - h_end + 1, stride):
            for _ in range(w_start, w - w_end + 1, stride):
                count += 1

    return count


def gen_patches(data_dir, num_patches, patch_size, stride):
    count = 0
    img_dir = os.path.join(data_dir, 'vi')

    patches = np.zeros((num_patches, 2, patch_size, patch_size),
                       dtype=np.uint8)

    for img in os.listdir(img_dir):
        img_path1 = os.path.join(img_dir, img)
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

        img_path2 = img_path1.replace('vi', 'po')
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        h, w = img1.shape

        # h_start = (h - patch_size) % stride // 2
        h_start = stride // 2
        h_end = patch_size + h_start

        # w_start = (w - patch_size) % stride // 2
        w_start = stride // 2
        w_end = patch_size + w_start

        for y in range(h_start, h - h_end + 1, stride):
            for x in range(w_start, w - w_end + 1, stride):
                patches[count, 0] = img1[y:y + patch_size, x:x + patch_size]
                patches[count, 1] = img2[y:y + patch_size, x:x + patch_size]
                count += 1

    return patches


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    # 0. config
    patch_size = 224
    stride = 100
    batch_size = 16
    data_name = 'polar'

    root_dir = os.path.join(BASE_DIR, '..', '..', 'datasets', data_name)
    data_dir = os.path.join(root_dir, 'train')
    # data_dir = os.path.join(root_dir, 'valid')

    # 1. count patches
    count = count_patches(data_dir, patch_size, stride)

    num_patches = int(np.ceil(count / batch_size)) * batch_size

    print('num patches: {}, batch size: {}, num iters: {}'.format(
        num_patches, batch_size, num_patches // batch_size))

    # 2. generate patches
    patches = gen_patches(data_dir, num_patches, patch_size, stride)

    if count < num_patches:
        pad = num_patches - count
        patches[count:] = patches[:pad]

    print('num patches: {}, batch size: {}, num iters: {}'.format(
        len(patches), batch_size,
        len(patches) // batch_size))

    # 3. save patches
    save_dir = os.path.join(data_dir, 'patches')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'patch_{}.npy'.format(patch_size)), patches)
    print('saving patches successfully')
