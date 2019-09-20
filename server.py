# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/10

import re
import os
import json

"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""
from utils.server_utils import log_server

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