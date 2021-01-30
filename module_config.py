# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2020/2/24
# @Author: JW Junjie_DC@foxmail.com

"""
由于注册机制会导致全局加载，有些后处理可能要特殊的环境，导致加载时就出现问题。
这个config本来应该是写在各个方法的config里面。但在导包时就执行的代码无法控制
提前设置一个全局变量指定哪些要屏蔽的
"""

# 白名单的中的表示only,（白名单为空时，表示不起作用）， 黑名单中的需要过滤（同理为空时表示不起作用）

black_module = []

# 只放行的名单
# used_module = []
white_module = ['vgg.py', 'image_line2_dataset.py',
               'text_detection_evaluation.py', 'centre_line_process.py']



