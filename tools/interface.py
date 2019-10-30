# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/1

import numpy as np
import torch
from core import build_model
from core.post_process import build_post_process
from core.pre_process import build_pre_process


"""
A helper class for inference
"""


class InferModel:
    def __init__(self, args):
        """
        :param args: the Config dict
        """
        self.model = build_model(args)
        self.model.setup(args)
        self.model.eval()
        self.post_process = build_post_process(args.post_process)
        self.pre_process = build_pre_process(args.pre_process)

    def expand_img(self, src):
        # todo: magic number
        # 与此同时，需要将原始的图片传出以供后处理
        if isinstance(src, np.ndarray):
            if len(src.shape) == 2:
                # 灰度图
                src = src[:, :, np.newaxis]
            src = src.transpose((2, 0, 1)).astype(np.float32)  # 通道前置
            src = torch.from_numpy(src)
        if isinstance(src, torch.Tensor):
            if len(src.shape) == 3:
                src = src.unsqueeze(dim=0)
            assert len(src.shape) == 4
        return src

    def infer(self, src, **args):
        src = self.pre_process(src)
        src_t = self.expand_img(src)
        self.model.set_input(src_t)
        self.model.test()
        out_dict = self.model.get_current_visuals()  # 可以有多个输出，比如attention的中间结果都可以输出
        # 后处理会有用到原始的图片吗？其实用不到，但是在调试过程中可能用到. 先也传入. 可能比如C2TD的校正.
        # todo: 可以通过args不定长参数来处理
        return self.post_process(out_dict, src, **args)


