# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/10/17
"""
图像数据处理相关的函数
"""
"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""
import cv2
import time
import requests
import base64
import numpy as np
from .check_base64 import is_base64





def img_size_process(input_img, input_type, max_pixel_sum, log_server):
    """
    根据图像的大小进行处理
    :param input_img: 三通道图像BGR
    :param input_type: 是否是原卷切割
    """
    code = 0

    # 区分原卷切割的标记
    if input_type == "paper_segment":
        log_server.logging('>>>>>>>> [Type] paper_segment')
    else:
        log_server.logging('>>>>>>>> [Type] other')

    h, w = input_img.shape[:2]
    # if h*w < 1600:
    #    code, message = 6, "fail"   # input image is too small
    #    return code, message

    if input_type == "paper_segment":
        if h * w > max_pixel_sum:  # 300w = 2000*1500
            code, message = 7, "fail"  # input image is too big
            return code, message
        else:
            return 0, (input_img, 1.0)
    else:
        re_img, re_scale = resize_image(input_img, max_w, max_h)
        # re_img, re_scale = input_img, 1.0
    return code, (re_img, re_scale)

def resize_image(input_img, max_w=2500, max_h=1800):
    """
    等比例缩放
    保证输入图片的宽度不大于max_w, 高度不大于max_h.
    """
    im_h, im_w = input_img.shape[:2]

    # 计算scale_w
    if im_w > max_w:
        scale_w = max_w / im_w
    else:
        scale_w = 1.0

    # 计算scale_h
    if im_h * scale_w > max_h:
        scale_h = max_h / (im_h * scale_w)
    else:
        scale_h = 1.0

    # 计算scale
    scale = scale_h * scale_w

    re_im = cv2.resize(input_img, (0, 0), fx=scale, fy=scale)

    return re_im, scale

def read_image_with_url(image_url, log_server):
    """
    image url -> numpy.array, [BGR]
    :param image_url: url of image
    :return:
        code: 返回码, 0表正常,非0表异常
        color_img: color image with type numpy.array
    """
    code = 0
    try:
        s1 = time.time()
        response = requests.get(image_url)
        s2 = time.time()
        log_server.logging('>>>>>>>> Time of io.imread: {:.2f}'.format(s2 - s1))
    except Exception as e:
        print(e)
        code, message = 3, "fail"
        return code, message
    if response.status_code != 200:
        code, message = 3, "fail"
        return code, message

    img_bytes = response.content
    im_base64 = base64.b64encode(img_bytes)
    im_base64 = str(im_base64, encoding="utf-8")

    code, color_img = decode_image_with_base64(im_base64)
    return code, color_img

def decode_image_with_base64(im_str, log_server, _type=cv2.IMREAD_COLOR):
    """
    从base64字符串解码为numpy.array
    :param base64_str: base64字符串
    :return:
        code: 返回码, 0表正常,非0表异常
        color_img: 彩色图片,numpy.array格式
    """
    code = 0

    b64_flag = is_base64(im_str)  # 判断输入字符串合法性
    if not b64_flag:
        code, message = 4, "fail"
        return code, message

    s1 = time.time()
    im_byte = base64.b64decode(im_str)
    im_arr = np.fromstring(im_byte, np.uint8)
    color_img = cv2.imdecode(im_arr, _type)
    # color_img = cv2.imdecode(im_arr, cv2.IMREAD_GRAYSCALE)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of decode base64 string: {:.2f}'.format(s2 - s1))

    if color_img is None:
        code, message = 5, "fail"
        return code, message
    log_server.logging('>>>>>>>> [DONE] Get Image')

    return code, color_img


