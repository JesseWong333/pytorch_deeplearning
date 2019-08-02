# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/1

import numpy as np
import torch
from core import build_model
from core.post_process import build_post_process


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

    def infer(self, src):
        # todo: magic number
        if isinstance(src, np.ndarray):
            if len(src.shape) == 2:
                # 灰度图
                src_ori = src[:, :, np.newaxis]
            src = src_ori.transpose((2, 0, 1)).astype(np.float32)  # 通道前置
            src = torch.from_numpy(src)
        if len(src.shape) == 3:
            src = src.unsqueeze(dim=0)
        assert len(src.shape) == 4
        self.model.set_input(src)
        self.model.test()
        out_dict = self.model.get_current_visuals()  # 可以有多个输出，比如attention的中间结果都可以输出
        return self.post_process(out_dict, src_ori)  # 后处理会有用到原始的图片吗？其实用不到，但是在调试过程中可能用到. 先也传入


