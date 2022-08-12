# Polarization Image Fusion with PFNet

This project is for personal study and under development, please click 'watch' or 'star' my repo and check back later if you are interested in it.

## 1 任务简述

主要任务：实现对强度图像和偏振图像的融合。

## 2 模型

本项目基于无监督的PFNet算法构建模型。

论文：https://doi.org/10.1364/ol.384189

代码：https://github.com/Junchao2018/Polarization-image-fusion

### 2.1 网络架构


### 2.2 损失函数

- 总损失

    ![](http://latex.codecogs.com/svg.latex?L_{total}(I_{S_0},I_{DoLP},I_f)=L_{mswssim}(I_{S_0},I_{DoLP},I_f)+\lambda\cdot%20L_{mae}(I_{avg},I_f))

- 结构损失：MSWSSIM Loss

    ![](http://latex.codecogs.com/svg.latex?SSIM(I_{S_0},I_{DoLP},I_f;w)=\frac{1}{N}\cdot\sum_{x,y}[\gamma\cdot%20SSIM(I_{S_0},I_f;w)+(1-\gamma)\cdot%20SSIM(I_{DoLP},I_f;w)])

    ![](http://latex.codecogs.com/svg.latex?L_{mswssim}(I_{S_0},I_{DoLP},I_f)=1-\frac{1}{5}\cdot\sum_{w\in[3,5,7,9,11]}SSIM(I_{S_0},I_{DoLP},I_f;w))

- 强度损失：MAE Loss

    ![](http://latex.codecogs.com/svg.latex?I_{avg}=\frac{1}{2}\cdot(I_{S_0}+I_{DoLP}))

    ![](http://latex.codecogs.com/svg.latex?L_{mae}(I_{avg},I_f)=\frac{1}{N}\cdot\sum_{x,y}\|\|I_{avg}-I_f\|\|_1)


## 3 数据集

本项目基于自建的偏振数据集训练模型。

统计信息：

- 训练集图片数：200 对

- 测试集图片数：32 对

- 图片分辨率：1224x1024

- 图片宽高比：1.2:1

## 4 评价指标

本项目基于主观性和客观性指标评价算法性能。

### 4.1 主观性评价


### 4.2 客观性评价

