import torch
import itertools
from .losses import build_loss
from .backbones import build_backbone
from .base_model import BaseModel
from . import register_model

"""
For more details about C2TD, see http://wiki.iyunxiao.com/display/guangzhouyanfazhongxin/C2TD+paper+draft
"""


@register_model
class C2TDModel(BaseModel):
    def name(self):
        return "C2TDModel"

    def initialize(self, args):
        BaseModel.initialize(self, args)
        self.loss_names = ['losses', 'total_loss']  # 需要沿用base_model的打印的机制，万一像GAN那样有很复杂的loss就不适用了
        self.model_names = ['net', ]
        self.net = build_backbone(args.backbone)

        # move net to GPU
        self.net.to(args.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, args.gpu_ids)


        self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()),
                                          lr=args.lr)
        # 初始化损失函数, 检测的网络通常有很复杂的loss
        self.criterion = build_loss(args.loss)

        # 会调用父类的setup方法为optimizers设置lr schedulers
        self.optimizers = []
        self.optimizers.append(self.optimizer)

    def set_input(self, input,):
        self.batch_images = torch.FloatTensor(input[0]).to(self.device)
        # self.label = input[1]  # 这个是一个list的
        self.heat_maps = torch.FloatTensor(input[1]).to(self.device)

    def forward(self):
        # 前向的写法， 可以写一些很复杂的前向途径, 比如多个模型组合，可以在这里自定义写一个前向传播
        self.score = self.net(self.batch_images)

    def backward(self):
        self.losses = self.criterion(self.heat_maps, self.score.permute(0, 2, 3, 1))
        if len(self.loss) <= 1:
            self.total_loss = self.losses

        self.total_loss = 0.
        for loss, ratio in zip(self.losses, self.arg.loss_ratios):
            self.total_loss += loss*ratio
        self.total_loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()