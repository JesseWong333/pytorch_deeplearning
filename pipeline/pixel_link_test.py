# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/19

import os
import cv2
import numpy as np
import torch
from tools import InferModel
from configs.config_util import ConfigDict


# todo: 标准的预处理过程抽象统一配置
def image_normalize(img, pixel_mean, std):
    img = img.astype(np.float32)
    img /= 255
    h, w, c = img.shape
    for i in range(c):
        img[:, :, i] -= pixel_mean[i]
        img[:, :, i] /= std[i]
    return img


def maybe_resize(img):
    # src_img = input.copy()
    ratio = 1
    if max(img.shape[:2]) > 3200:
        if img.shape[0] > img.shape[1]:
            ratio = 3200 / img.shape[0]
        else:
            ratio = 3200 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img, ratio


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


pixel_mean = [0.91610104, 0.91612387, 0.91603917]
std_mean = [0.2469206, 0.24691878, 0.24692364]

if __name__ == '__main__':
    from configs.pixel_link import config
    args = ConfigDict(config)
    args.isTrain = False

    img_base_path = '/media/Data/hcn/data/pixellink_data/'

    c2td_model = InferModel(args)

    for img, file_name in img_generator(img_base_path):
        h, w, _ = img.shape
        img, ratio = maybe_resize(img)
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        img = image_normalize(img, pixel_mean, std_mean)
        d = c2td_model.infer(img, ratio=ratio, src_img_shape=img.shape)
        print(d)

