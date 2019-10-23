# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/10


import re
import os
import json
import time
import requests
import base64
from .logserver import LogServer
from .images_utils import read_image_with_url, decode_image_with_base64

"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""


def get_gpu_use():
    """
    查看当前进程的GPU使用率(MiB)
    """
    curr_pid = str(os.getpid())
    lines = os.popen('nvidia-smi').readlines()
    lines = lines[::-1]
    mem_use = 0
    for line in lines:
        if curr_pid in line and "python" in line:
            line_pid = re.split(" +", line)[2]
            if curr_pid == line_pid:
                res = re.findall("{}.* (.*?)MiB".format(curr_pid), line)
                if res:
                    mem_use = int(res[0])
                    break
    return mem_use


def parse_post_data(post_data):
    """
    解释服务接收到的数据
    """
    code = 0
    if not post_data:
        code, message = 1, "fail"
        return code, message
    if isinstance(post_data, bytes):     # bytes->str
        post_data = post_data.decode()
    try:
        post_data = json.loads(post_data)     # 通用字符串转为字典
    except:
        # "Input format is incorrect!"
        code, message = 2, "fail"
        return code, message
    return code, post_data

def get_data(request, log_server):
    data = request.data  # 获取post的数据

    code = 0
    # 判断输入数据的合法性
    code, message = parse_post_data(data)
    if code:
        return code, message

    data = message

    im_url = data.get("im_url")  # 图片url
    subject = data.get("subject")  # 科目
    _type = data.get("type")  # 原卷切割标记
    from_src = data.get("from_src")  # 调用用户标记

    im_str = None
    if im_url:  # 如果图片url存在，优先按照图片url方式调用服务
        log_server.logging('>>>>>>>> [START] Get Image From: %s' % im_url)
        log_server.logging('>>>>>>>> from: %s' % from_src)

        code, message = read_image_with_url(im_url)
    else:
        log_server.logging('>>>>>>>> [START] Get Image from base64')
        log_server.logging('>>>>>>>> from: %s' % from_src)
        im_str = request.values.get("im_base64")
        if im_str is None:
            im_str = data.get("im_base64")

        code, message = decode_image_with_base64(im_str)

    if code:  # 读取图片错误或解码错误
        return code, message

    color_img = message  # 读图成功的情况下, 第二个参数为图片

    # ret_data = (subject, _type, from_src, color_img)
    ret_data = (im_url, im_str, subject, _type, from_src, color_img)

    return code, ret_data





