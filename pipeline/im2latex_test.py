# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/20

import os
import cv2
import numpy as np
import torch
from tools import InferModel
from configs.config_util import ConfigDict, load_config
from PIL import Image
from torchvision import transforms


img_path = '/media/Data/hzc/datasets/formulas/all_cleaned/1770829_0/1770829_0_6.png'
images = []
formulas = []
image = transforms.ToTensor()(Image.fromarray(cv2.imread(img_path, 0)))
img = np.ones((image.shape[0], image.shape[1], image.shape[2]))
img[:, 0:image.shape[1], 0:image.shape[2]] = image
images.append(img[np.newaxis, :, :, :])
images = np.concatenate(images, axis=0)
images = images.astype(np.float32)
formulas.append('')
input_data = (images, formulas)

if __name__ == '__main__':
    # todo: 原先的im2latex部分参数通过yaml配置, 并没有整合
    from configs.im2latex import config
    args = ConfigDict(config)
    args.isTrain = False
    im2latex_args = ConfigDict(load_config(args.im2latex_congigs))
    args.update(im2latex_args)

    im2latex_model = InferModel(args)

    d = im2latex_model.infer(input_data)
    print(d)
