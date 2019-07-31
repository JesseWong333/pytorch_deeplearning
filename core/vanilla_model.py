# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/18

import torch
import itertools
from core.losses import build_loss
from .backbones import build_backbone
from .base_model import BaseModel
from . import register_model


@register_model
class VanillaModel(BaseModel):
    def name(self):
        return "VanillaModel"

    def initialize(self, args):
        BaseModel.initialize(self, args)
        # 在这里初始化模型, 初始化optimizer,
        # 网络的train() 和 eval() 模式可以在这里切换
        self.loss_names = ['loss', ]  # 在这里指定需要打印的loss名字， 这个名字需要跟loss的属性名字完全一致
        self.model_names = ['net', ]  # 在这里指定网络模型的名字，这个对象的一个属性
        self.net = build_backbone(args)

        # move net to GPU
        self.net.to(self.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, self.gpu_ids)

        self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()),
                                          lr=args.lr)

        self.criterion = build_loss(args.loss)

        # 会调用父类的setup方法为optimizers设置lr schedulers # todo: lr schedulers方法的抽象
        self.optimizers = []
        self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.src = torch.FloatTensor(input[0]).to(self.device)
        self.target = torch.FloatTensor(input[1]).to(self.device)

    def forward(self):
        self.score = self.net(self.src)

    def backward(self):
        # 如果有多个loss呢? 怎么抽象, 不同的loss， 如何加权
        self.loss = self.criterion(self.target, self.score)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()