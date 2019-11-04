# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/18

import torch
import itertools
from core.losses import build_loss
from .backbones import build_backbone
from .base_model import BaseModel
from . import register_model
from .post_process import build_post_process


@register_model
class RecognitionModel(BaseModel):
    def initialize(self, args):
        BaseModel.initialize(self, args)
        if args.get('loss_ratios') is None:
            self.loss_names = ['total_loss', ]
        else:
            self.loss_names = ['total_loss', 'losses', ]
        self.model_names = ['net', ]
        self.output_names = ['pred', ]
        self.net = build_backbone(args.backbone)

        # move net to GPU
        self.net.to(args.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, args.gpu_ids)

        if args.isTrain:
            self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()),
                                              lr=args.lr)

            self.criterion = build_loss(args.loss)
            self.optimizers = []
            self.optimizers.append(self.optimizer)
        # else:
        # if args.require_evaluation:
        self.post_process = build_post_process(args.post_process)

    def set_input(self, input,):
        # todo: a better way to design the input
        # print(input[1].dtype, input[1].shape)
        if self.args.isTrain:
            self.src = torch.FloatTensor(input[0]).to(self.device)
            self.text = torch.IntTensor(input[1]).to(self.device)
            # print(torch.min(self.text))
            self.length = torch.IntTensor(input[2]).to(self.device)
        else:
            self.src = torch.FloatTensor(input).to(self.device)

    def forward(self):
        self.pred = self.net(self.src)

    def backward(self):
        self.losses = self.criterion(self.text, self.length, self.pred)
        if not isinstance(self.losses, tuple):
            self.total_loss = self.losses
        else:
            self.total_loss = 0.
            for loss, ratio in zip(self.losses, self.args.loss_ratios):
                self.total_loss += loss*ratio
        self.total_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    # todo, 添加评估方法
    def evaluate(self):
        self.forward()
        loss = self.criterion(self.text, self.length, self.pred)
        if not isinstance(self.losses, tuple):
            self.total_loss = self.losses
        else:
            self.total_loss = 0.
            for loss, ratio in zip(self.losses, self.args.loss_ratios):
                self.total_loss += loss * ratio
        out = self.post_process(self.pred)
        return out, loss


