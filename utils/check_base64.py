# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/10/17
"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""
import string


def is_base64(input_string):
    '''
    检查输入字符串是否为base64编码
    :param input_string:
    '''
    if not input_string:    # 空
        return False

    if not isinstance(input_string, str):    # 确保类型
        return False

    if len(input_string) % 4 != 0:        # 能被4整除
        return False
    # base64的字符必定属于[0-9,a-z,A-Z,+,/,=]
    b64_char_set = string.ascii_letters+string.digits+"+/="

    b64_flag = True
    for input_char in input_string:
        if input_char not in b64_char_set:
            b64_flag = False
            break

    return b64_flag
