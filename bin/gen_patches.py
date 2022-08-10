# -*- coding: utf-8 -*-
"""
# @file name  : gen_patches.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-08-05
# @brief      : 图像块生成
"""

import os

import cv2
import numpy as np


def count_patches(root_path, patch_size, stride):
    count = 0
    vis_dir = os.path.join(root_path, 'vis')

    for img in os.listdir(vis_dir):
        vis_path = os.path.join(vis_dir, img)
        vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

        h, w = vis_img.shape

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


def gen_patches(root_path, num_patches, patch_size, stride):
    count = 0
    vis_dir = os.path.join(root_path, 'vis')

    patches = np.zeros((num_patches, 2, patch_size, patch_size),
                       dtype=np.uint8)

    for img in os.listdir(vis_dir):
        vis_path = os.path.join(vis_dir, img)
        vis_img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

        dolp_path = vis_path.replace('vis', 'dolp')
        dolp_img = cv2.imread(dolp_path, cv2.IMREAD_GRAYSCALE)

        h, w = vis_img.shape

        # h_start = (h - patch_size) % stride // 2
        h_start = stride // 2
        h_end = patch_size + h_start

        # w_start = (w - patch_size) % stride // 2
        w_start = stride // 2
        w_end = patch_size + w_start

        for y in range(h_start, h - h_end + 1, stride):
            for x in range(w_start, w - w_end + 1, stride):
                patches[count, 0] = vis_img[y:y + patch_size, x:x + patch_size]
                patches[count, 1] = dolp_img[y:y + patch_size,
                                             x:x + patch_size]
                count += 1

    return patches


if __name__ == '__main__':

    import os
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    # 0. config
    root_path = os.path.join('..', '..', 'data', 'train')
    # root_path = os.path.join('..', '..', 'data', 'valid')

    patch_size = 224
    stride = 100
    batch_size = 16

    # 1. count patches
    count = count_patches(root_path, patch_size, stride)

    num_patches = int(np.ceil(count / batch_size)) * batch_size

    print('num patches: {}, batch size: {}, num iters: {}'.format(
        num_patches, batch_size, num_patches // batch_size))

    # 2. generate patches
    patches = gen_patches(root_path, num_patches, patch_size, stride)

    if count < num_patches:
        pad = num_patches - count
        patches[count:] = patches[:pad]

    print('num patches: {}, batch size: {}, num iters: {}'.format(
        patches.shape[0], batch_size, patches.shape[0] // batch_size))

    # 3. save patches
    save_path = os.path.join(root_path, 'patches')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, 'patch_{}.npy'.format(patch_size)),
            patches)
    print('saving patches successfully')
