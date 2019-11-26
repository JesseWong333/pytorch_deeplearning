# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/2
import sys
sys.path.append("..")
import os
import cv2
import numpy as np
import torchvision.transforms.functional as F
import operator
from PIL import ImageDraw, Image, ImageFont
from tools import InferModel
from utils.bboxes_utils import process_bboxes
from utils.images_utils import maybe_resize
from utils.skew_correct import correct_image
from utils.bboxes_utils import sort_boxes
import glob
from utils.config_util import ConfigDict

"""
检测 + 识别 pipeline
pipeline 要抽象吗？ 使用多线程机制
用于连接起来测试
-- 这样写完全还是一团糟~~
"""

class OCRPipeLine(object):
    def __init__(self, ):
        #模型配置参数是在这里导入好，还是放在外面，当成参数传进来？
        from configs.c2td import config as c2td_config
        from configs.resbilstm import config as resbilstm_config

        c2td_args = ConfigDict(c2td_config)
        c2td_args.isTrain = False
        resbilstm_args = ConfigDict(resbilstm_config)
        resbilstm_args.isTrain = False

        self.det_model = InferModel(c2td_args) # 检测模型
        self.reg_model = InferModel(resbilstm_args) #识别模型

    def one_image_reg(self, img, boxes, subject):
        labellist = []
        for box in boxes:
            imgcrop = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            label = self.reg_model.infer(imgcrop, subject=subject)
            labellist.append(label)
        return labellist

    def infer(self, img):#单个例子的
        src_img = img.copy()
        img, ratio = maybe_resize(img, 3000) #对图片做resize
        boxes_coord = self.det_model.infer(img, ratio=ratio, src_img_shape=src_img.shape)  # 如果是扭曲的，这里应该是返回校正后的图片，为了接口统一，检测统一返回校正后的图片
        labellist = []
        new_boxs = []
        # print(boxes_coord)
        for box in boxes_coord:  # 是否可以并行
            imgcrop = src_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            label = self.reg_model.infer(imgcrop, subject='历史')
            # if len(label) > 0:
            labellist.append(label)
                # new_boxs.append(box)


        #对方框进行排序
        boxes_coord = np.asarray(boxes_coord, dtype=int).tolist()

        # 检测结果排序第二版本
        boxes_coord_t = sort_boxes(boxes_coord)
        sort_ids = [boxes_coord.index(box_t) for box_t in boxes_coord_t]
        labellist_t = np.array(labellist)[sort_ids]
        labellist_t = labellist_t.tolist()


        return boxes_coord_t, labellist_t


if __name__ == '__main__':
    from tqdm import tqdm

    def img_generator(folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            file_name = img_path.split('/')[-1].split('.')[0]
            yield img, file_name

    pipeline = OCRPipeLine()

    img_base_path = '/media/Data/hcn/data/C2TD_TEST/problem'
    save_path = '/media/Data/hzc/code/pytorch_deeplearning/tmp_1126'

    imgpaths = glob.glob(os.path.join(img_base_path, '*.png'))

    for img_path in tqdm(imgpaths):
        # print(file_name)
        basename = os.path.basename(img_path)
        txtname = os.path.join(save_path, basename.split('.', 1)[0] + '.txt')
        if os.path.exists(txtname):
            continue
        img = cv2.imread(img_path)
        img, _ = correct_image(img)
        cv2.imwrite(os.path.join(save_path, basename), img)
        if max(img.shape) > 3000:
            print(img.shape)
        bounding_boxes, texts = pipeline.infer(img)
        txtfile = open(txtname, 'w', encoding='utf-8')
        for i in range(len(bounding_boxes)):
            txtfile.write('{}, {}, {}, {}, {}\n'.format(int(bounding_boxes[i][0]), int(bounding_boxes[i][1]), int(bounding_boxes[i][2]), int(bounding_boxes[i][3]), texts[i]))
        txtfile.close()

        # visualize(img, bounding_boxes, texts, img_patches, font, file_name, save_path)

