# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/19
import sys
sys.path.append("..")
import os
import cv2
import numpy as np
import torch
from tools import InferModel
from utils.config_util import ConfigDict
import glob


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
    if max(img.shape[:2]) > 960:
        if img.shape[0] > img.shape[1]:
            ratio = 960 / img.shape[0]
        else:
            ratio = 960 / img.shape[1]
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
    c2td_model = InferModel(args)

    datadir = r'/home/chen/mcm/ctpn_test_2019_01_23/test_data_all_latest/high_math'
    vis_path = r'/home/chen/hcn/data/pixel_link_data/formula_detection/high_math_1029'
    img_paths = glob.glob(os.path.join(datadir, '*.png'))
    for img_path in img_paths:
        print(img_path)
        img = cv2.imread(img_path)
        img_copy = img.copy()
        img_shape = img.shape
        img, ratio = maybe_resize(img)
        # img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        # img = image_normalize(img, pixel_mean, std_mean)
        cords, pixel_pos_scores, link_pos_scores, mask = c2td_model.infer(img, ratio=ratio, src_img_shape=img_shape)
        # hot_map = np.squeeze(np.where(pixel_pos_scores >= 0.7, 255, 0))
        # link_pos_scores = np.squeeze(link_pos_scores, 0)
        # link_mask_0 = np.where(link_pos_scores[:, :, 0] >= 0.5, 255, 0)
        # link_mask_1 = np.where(link_pos_scores[:, :, 1] >= 0.5, 255, 0)
        # link_mask_2 = np.where(link_pos_scores[:, :, 2] >= 0.5, 255, 0)
        # link_mask_3 = np.where(link_pos_scores[:, :, 3] >= 0.5, 255, 0)
        for cord in cords:
            cv2.rectangle(img_copy, (cord[0], cord[1]), (cord[4], cord[5]), [0, 0, 255], 2, 4)
        basename = os.path.basename(img_path)
        savename = os.path.join(vis_path, basename)
        # hotmap_name = os.path.join(vis_path, basename.split('.', 1)[0] + '_scoremap.png')
        # link_0 = os.path.join(vis_path, basename.split('.', 1)[0] + '_0.png')
        # link_1 = os.path.join(vis_path, basename.split('.', 1)[0] + '_1.png')
        # link_2 = os.path.join(vis_path, basename.split('.', 1)[0] + '_2.png')
        # link_3 = os.path.join(vis_path, basename.split('.', 1)[0] + '_3.png')
        # mask_name = os.path.join(vis_path, basename.split('.', 1)[0] + '_mask.png')
        # mask = np.where(mask==1, 255, 0)
        cv2.imwrite(savename, img_copy)
        # cv2.imwrite(hotmap_name, hot_map)
        # cv2.imwrite(link_0, link_mask_0)
        # cv2.imwrite(link_1, link_mask_1)
        # cv2.imwrite(link_2, link_mask_2)
        # cv2.imwrite(link_3, link_mask_3)
        # cv2.imwrite(mask_name, mask)




