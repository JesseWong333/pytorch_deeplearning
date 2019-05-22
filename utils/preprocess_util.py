# coding: utf-8

from __future__ import print_function
import sys
import os

sys.path.append(os.getcwd())


import cv2
from scipy.stats import entropy
import numpy as np


def rotate_bound(image, rotate_angle, pad_value=(0, 0, 0)):
    """
    旋转图片，超出原图大小的边界用指定颜色填充.
    :param image: 待旋转图片
    :param rotate_angle:  旋转角度
    :param pad_value: 边界填充颜色
    :return:
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rotate_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=pad_value)


def correct_image(src_image):
    """
    倾斜校正.
    :param src_image: 待校正图片
    :return:
    """
    if src_image.ndim == 3:
        gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = src_image

    _, bin_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    thetas = np.linspace(-2, 2., 50, endpoint=False)
    entropys = []
    for angle in thetas:
        rotated_image = rotate_bound(bin_image, angle)

        sum_row = np.sum(rotated_image, axis=1)
        sum_row = sum_row / float(np.sum(sum_row))
        entropys.append(entropy(sum_row))

    angle = thetas[np.argmin(entropys)]

    # (-00, -0.2)U(0.2, +00)范围内旋转, [0, 0.2]不旋转
    if abs(angle) > 0.2:
        ret_image = rotate_bound(src_image, angle, (255, 255, 255))
        return ret_image, angle

    return src_image, angle


def resize_im(im, scale, max_scale=None):
    """
    以最小边为scale比例缩放图片.
    """
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    """
    ctpn结果可视化
    """
    base_name = image_name.split('/')[-1]
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if box[2] - box[0] < 5 or box[5] - box[1] < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            box_int = list(map(lambda x: int(x), box[:8]))
            cv2.line(img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color, 2)
            cv2.line(img, (box_int[0], box_int[1]), (box_int[4], box_int[5]), color, 2)
            cv2.line(img, (box_int[6], box_int[7]), (box_int[2], box_int[3]), color, 2)
            cv2.line(img, (box_int[4], box_int[5]), (box_int[6], box_int[7]), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale),
                        int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale),
                        int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale),
                        int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale),
                        int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale)
    base_name = base_name.replace("png", "jpg")
    cv2.imwrite(os.path.join("data/results", base_name), img)

