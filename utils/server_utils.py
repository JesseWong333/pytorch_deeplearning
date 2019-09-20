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


def read_image_with_url(image_url):
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