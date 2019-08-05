# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/2

"""
检测 + 识别 pipeline
pipeline 要抽象吗？ 使用多线程机制
用于连接起来测试
"""

import os
import cv2
import numpy as np
import torchvision.transforms.functional as F
from tools import InferModel
from configs.config_util import ConfigDict


# from mcm
def visualize(im_name, boxes_coord, labellist, out_dir, font, scale=1.0, subject=None):
    """
    检测与识别结果可视化.
    :param im_name: 图片路径/图片url
    :param boxes_coord: 检测框位置信息[[xmin, ymin, xmax, ymax],...]
    :param labellist: 检测框识别结果字符串[res_string1, res_string2]
    :param out_dir: 结果写出目录
    :param font: 字体类型及大小
    :param scale: 缩放比例
    :param subject: 科目类别
    """
    if not boxes_coord:  # 保证检测框个数>=1
        return
    boxes_coord = np.asarray(boxes_coord, dtype=int)/scale  # 缩放到原图的比例坐标
    boxes_coord = boxes_coord.astype(int)
    boxes_coord = boxes_coord.tolist()


class PipeLine(object):
    def __init__(self):
        from configs.c2td import config as c2td_config
        from configs.crnn import config as crnn_config
        c2td_args = ConfigDict(c2td_config)
        c2td_args.isTrain = False
        crnn_args = ConfigDict(crnn_config)
        crnn_args.isTrain = False

        self.c2td_model = InferModel(c2td_args)
        self.crnn_model = InferModel(crnn_args)

    def infer(self, img):
        h, w, _ = img.shape
        # 检测填充
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        img_patches = self.c2td_model.infer(img)  # 如果是扭曲的，这里应该是返回校正后的图片，为了接口统一，检测统一返回校正后的图片

        nh = 32  # todo magic number
        for img_patch in img_patches:  # 是否可以并行
            # 这个版本的crnn要求高度为32的灰度图, 且需要进行输入图片的归一化
            gray_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            h, w = gray_patch.shape
            nw = int(nh * w / h)
            gray_patch = cv2.resize(gray_patch, (nw, nh))
            gray_patch = np.expand_dims(gray_patch, axis=2)
            gray_patch = F.to_tensor(gray_patch)  # 这个标准步骤我以后还是做一下比较好，进行数据归一化
            gray_patch.sub_(0.5).div_(0.5)
            text = self.crnn_model.infer(gray_patch)
        pass


if __name__ == '__main__':
    def img_generator(folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            file_name = img_path.split('/')[-1].split('.')[0]
            yield img, file_name

    pipeline = PipeLine()

    img_base_path = '/media/Data/wangjunjie_code/advancedEast/dataset/junior_biology'

    for img, file_name in img_generator(img_base_path):
        pipeline.infer(img)
        pass


