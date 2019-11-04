# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/19
import torch.nn.functional as F
import numpy as np
import cv2
from . import register_post_process
from utils.config_util import ConfigDict
from configs.pixel_link import config as pixel_link_config
config = ConfigDict(pixel_link_config['pixel_link'])


"""The codes are form https://git.iyunxiao.com/DeepVision/pytorch-ocr-framework"""


@register_post_process
def pixel_link_process(outputs, img, ratio, src_img_shape):
    score = outputs[0]

    shape = score.shape
    neighbor_num = int((shape[1] - 2) / 2)
    pixel_pos_scores = F.softmax(score[:, 0:2, :, :], dim=1)[:, 1, :, :]
    # pixel_pos_scores=torch.sigmoid(outputs[:,1,:,:])
    # FIXME the dimention should be changed
    link_scores = score[:, 2:, :, :].view(shape[0], 2, neighbor_num, shape[2], shape[3])
    link_pos_scores = F.softmax(link_scores, dim=1)[:, 1, :, :, :]
    # print(cords)
    pixel_pos_scores = pixel_pos_scores.cpu().numpy()
    link_pos_scores = link_pos_scores.cpu().numpy()

    """ ratio 是根据已有的图片的缩放比例动态调整 """

    return process(pixel_pos_scores, link_pos_scores, shape[2:], ratio, src_img_shape,)


def process(pixel_pos_scores, link_pos_scores, img_shape, ratio, src_img_shape):

    link_pos_scores = np.transpose(link_pos_scores, (0, 2, 3, 1))

    mask = decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = mask_to_bboxes(mask, img_shape)
    cords = np.array(bboxes, dtype=np.float32)
    cords *= 4
    cords /= ratio
    cords = cords.astype(np.int32)

    # copy_pixel_pos_scores = np.squeeze(pixel_pos_scores)
    # copy_pixel_pos_scores = np.where(copy_pixel_pos_scores > 0.5, 255, 0)
    # savepath = os.path.join(vis_path, file_name + '_scoremap.png')
    # cv2.imwrite(savepath, copy_pixel_pos_scores)
    final_cords = []

    if len(bboxes) > 0:
        cords[:, 0::2] = np.clip(cords[:, 0::2], 0, src_img_shape[1] - 1)
        cords[:, 1::2] = np.clip(cords[:, 1::2], 0, src_img_shape[0] - 1)
        # final_cords = bboxes_to_quard(cords)

    return cords, pixel_pos_scores, link_pos_scores, mask


def bboxes_to_quard(bboxes):
    """
    将4点坐标方框转换为左上、右下坐标方框
    :param bboxes:
    :return:
    """
    cords = np.ones((bboxes.shape[0], 4), dtype=int)
    cords[:, 0] = bboxes[:, 0::2].min(axis=1)
    cords[:, 1] = bboxes[:, 1::2].min(axis=1)
    cords[:, 2] = bboxes[:, 0::2].max(axis=1)
    cords[:, 3] = bboxes[:, 1::2].max(axis=1)
    return cords

def decode_batch(pixel_cls_scores, pixel_link_scores,
                 pixel_conf_threshold=None, link_conf_threshold=None):

    if pixel_conf_threshold is None:
        pixel_conf_threshold = config.pixel_conf_threshold

    if link_conf_threshold is None:
        link_conf_threshold = config.link_conf_threshold


    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in range(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores,
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def decode_image(pixel_scores, link_scores,
                 pixel_conf_threshold, link_conf_threshold):
    if config.decode_method == "DECODE_METHOD_join":
        mask = decode_image_by_join(pixel_scores, link_scores,
                                    pixel_conf_threshold, link_conf_threshold)
        return mask
    # elif config.decode_method == "DECODE_METHOD_border_split":
    #     return decode_image_by_border(pixel_scores, link_scores,
    #                                   pixel_conf_threshold, link_conf_threshold)
    else:
        raise ValueError('Unknow decode method:%s' % (config.decode_method))


import pyximport
pyximport.install()
from .pixel_link_decode import decode_image_by_join


def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points


def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h


def mask_to_bboxes(mask, image_shape, min_area=None,
                   min_height=None, min_aspect_ratio=None):
    # feed_shape = config.train_image_shape

    image_h, image_w = image_shape[0:2]

    if min_area is None:
        min_area = config.min_area

    if min_height is None:
        min_height = config.min_height
    bboxes = []
    max_bbox_idx = mask.max()
    mask = cv2.resize(mask, (image_w, image_h),
                           interpolation=cv2.INTER_NEAREST)

    for bbox_idx in range(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
        #         if bbox_mask.sum() < 10:
        #             continue
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)

        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue

        if rect_area < min_area:
            continue

        #         if max(w, h) * 1.0 / min(w, h) < 2:
        #             continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)

    return bboxes


def find_contours(mask, method=None):
    """这个函数原来是dataset中的"""
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                   method=method)
    except:
        _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                  method=method)
    return contours
