# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/10/17
"""
图像数据处理相关的函数
"""
import sys
sys.path.append("..")
import cv2
import config
import time
import requests
import numpy as np
import base64
from .check_base64 import is_base64
import json
import io
from PIL import Image


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

def maybe_resize(img, max_len):
    # src_img = input.copy()
    ratio = 1
    if max(img.shape[:2]) > max_len:
        if img.shape[0] > img.shape[1]:
            ratio = max_len / img.shape[0]
        else:
            ratio = max_len / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
    return img, ratio

def img_size_process(input_img, input_type, log_server):
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
        if h * w > config.max_pixel_sum:  # 630w = 3000*2100
            code, message = 7, "fail：input image is too big during paper_segment"  # input image is too big
            return code, message
        else:
            return 0, (input_img, 1.0)
    else:
        re_img, re_scale = resize_image(input_img, config.max_w, config.max_h)
        # re_img, re_scale = input_img, 1.0
    return code, (re_img, re_scale)

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
        code, message = 3, "fail: the url of image is incorrect or not exists"
        return code, message
    if response.status_code != 200:
        code, message = 3, "fail: the url of image is incorrect or not exists"
        return code, message

    img_bytes = response.content
    im_base64 = base64.b64encode(img_bytes)
    im_base64 = str(im_base64, encoding="utf-8")

    code, color_img = decode_image_with_base64(im_base64)
    return code, color_img

def decode_image_with_base64(im_str, log_server,_type=cv2.IMREAD_COLOR):
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
        code, message = 4, "fail: the input data is not encoded by base64"
        return code, message

    s1 = time.time()
    im_byte = base64.b64decode(im_str)
    im_arr = np.fromstring(im_byte, np.uint8)
    color_img = cv2.imdecode(im_arr, _type)
    # color_img = cv2.imdecode(im_arr, cv2.IMREAD_GRAYSCALE)
    s2 = time.time()
    log_server.logging('>>>>>>>> Time of decode base64 string: {:.2f}'.format(s2 - s1))

    if color_img is None:
        code, message = 5, "fail: image decoding fail, maybe the image data is incorrect or incomplete"
        return code, message
    log_server.logging('>>>>>>>> [DONE] Get Image')

    return code, color_img

def cv_img2base64(cv_img):
    """
    way4: np.ndarray->PIL.Image-->二进制数据-->base64(str)
    :param im_path:
    :return:
    """
    im = Image.fromarray(cv_img)
    imByteArr = io.BytesIO()
    #im.save(imByteArr, format="JPEG")
    im.save(imByteArr, format="PNG")    # PNG不作压缩
    imByteArr = imByteArr.getvalue()

    im_base64 = base64.b64encode(imByteArr)
    im_base64 = str(im_base64, encoding="utf-8")

    return im_base64

#公式code结果及消息要不要返回？？？如果出问题了，可能不知道是出的什么问题
def formula_det_reco(input_color_img, log_server, from_src=None, subject=None):
    """
    :input_color_img: 输入的三通道图像, numpy.array, BGR
    :from_src: 来源
    :subject: 科目, 物理、化学、数学、生物之一
    """
    # 对数学学科进行公式识别
    dict_str = json.dumps({"im_base64": cv_img2base64(input_color_img), "from_src": from_src, "subject": subject})

    try:
        s1 = time.time()
        response = requests.post(config.formula_url, data=dict_str, timeout=10)
        s2 = time.time()
        log_server.logging('>>>>>>>> formula det rec time: {:.2f}'.format(s2 - s1))
    except Exception as e:
        print(e)
        # message = "formula recognize fail"
        return [], [], input_color_img, (8, 'formula recognize fail')

    ret_dict = json.loads(response.text)
    status = ret_dict.get("code")  # 3.判断响应状态是否正常
    if status:
        return [], [], input_color_img, (status, ret_dict.get("message"))

    # 4.返回检测结果，识别结果
    data = ret_dict.get("data")
    if not data:
        return [], [], input_color_img, (status, ret_dict.get("message"))
    data = json.loads(data)
    boxes_coord = data.get("boxes_coord")
    if not boxes_coord:
        # print("Nothing can be detected!")
        return [], [], input_color_img, (status, ret_dict.get("message"))
    labellist = data.get("labellist")  # 公式接口出来的结果会自带${}$

    # 公式区域填白
    output_color_img = input_color_img.copy()
    for box_coord in boxes_coord:
        xmin, ymin, xmax, ymax = box_coord
        output_color_img[ymin:ymax, xmin:xmax] = 255  # 填白

    return boxes_coord, labellist, output_color_img, (status, ret_dict.get("message"))