# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/28

import os
import cv2
from tools import InferModel
from utils.config_util import ConfigDict


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'jpg', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path, -1)
        file_name = img_path.split('/')[-1].split('.')[0]
        # w, h = image.size
        # h, w = img.shape
        # nh = 32
        # nw = int(nh * w / h)
        # img = cv2.resize(img, (nw, nh))
        # img = np.expand_dims(img, axis=2)
        # img = F.to_tensor(img)  # 这个标准步骤我以后还是做一下比较好，进行数据归一化
        # img.sub_(0.5).div_(0.5)
        yield img, file_name


if __name__ == '__main__':
    from configs.segmentation_unet import config
    args = ConfigDict(config)
    args.isTrain = False

    img_base_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/severstal-steel-defect-detection'

    infer_model = InferModel(args)

    for img, file_name in img_generator(img_base_path):
        d = infer_model.infer(img)
        pass
