# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/1

import numpy as np
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