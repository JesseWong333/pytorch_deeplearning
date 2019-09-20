# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/1

import numpy as np
import cv2
from . import register_post_process
"""
包含一些基本的后处理，比如：将tensor转换为
"""


@register_post_process
def convert_to_numpy(img_t):
    """
    convert tensor img to numpy img
    :param img_t:
    :return:
    """
    if img_t.is_cuda:
        img_t = img_t.cpu()
    img_t = img_t.permute(0, 3, 1, 2)
    return img_t.numpy().astype(np.int)


#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


@register_post_process
def convert_seg_to_encoded_pixels(img_t, thresh=0.9):
    """
    convert multiple segmentation map to encoded pixels. 4 channel
    """
    img_t = img_t[0]
    _, img_index = img_t.max(1)
    if img_index.is_cuda:
        img_index = img_index.cpu()

    # 分各个通道开始
    predictions = []
    mask_channel = np.zeros((256, 1600, 4), dtype=np.uint8)
    mask = img_index.squeeze(0).numpy()
    for c_class in range(4):
        mask_channel[mask == c_class + 1, c_class] = 1
        label = mask2rle(mask_channel[:, :, c_class])
        predictions.append(label)

    # img[img >= thresh] = 1.
    # img[img < thresh] = 0.
    # img = img * 255
    # h, w, _ = img.shape
    # encoded_pixels = [[], [], [], []]
    # for channel in range(4):
    #     _, bin = cv2.threshold(img[:, :, channel], 255 * thresh, 255, cv2.THRESH_BINARY)
    #     contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     for cnt in contours:
    #         cnt = np.squeeze(cnt, axis=1)
    #         sorted_cnt = cnt[np.lexsort(cnt[:, ::-1].T)]  # 应该是先按照x排序好， x相同的再按照y排序好
    #         start_x = sorted_cnt[0, 0]
    #         start_index = 0
    #         for i in range(1, sorted_cnt.shape[0]):
    #             if sorted_cnt[i, 0] != start_x:
    #                 number_pixels = sorted_cnt[i-1, 1] - sorted_cnt[start_index, 1] + 1
    #                 encoded_pixels[channel].append(h*start_x+sorted_cnt[start_index, 1])
    #                 encoded_pixels[channel].append(number_pixels)
    #
    #                 start_x = sorted_cnt[i, 0]
    #                 start_index = i
    return predictions
