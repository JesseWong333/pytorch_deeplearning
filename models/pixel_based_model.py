import torch
import itertools
from losses.centre_line_loss import CentreLineLoss
from .networks import VGGPixel
from .base_model import BaseModel

"""
检测中心线， 并且使用中心线去预测上下的边界
中心线使用hough的方式去找， 或者使用opencv的找轮廓. 不要自己去写聚类
使用“山峰”逐渐降低的方式去标记数据，而不是全部标记为1. 中心线通过梯度的方式去找一条线， 不要设置绝对的阈值 
"""


class PixelBasedModel(BaseModel):
    def name(self):
        return "PixelBasedModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # called at model.__init__.py then at base_option.py gather_options
        # 在这里写当前模型的参数
        parser.add_argument('--in_channels', default=3, type=int,
                            help='the in_channel of the image,  should be 1 or 3')
        parser.add_argument('--channel_width', type=float, default=1.0,
                            help='the channel width of the convolution layer. The out channel of each conv layer '
                                 'except for the last layer will multiply this float ')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # 在这里初始化模型, 初始化optimizer,
        # 网络的train() 和 eval() 模式可以在这里切换
        self.loss_names = ['loss', 'heatmap_loss', 'location_loss']  # 在这里指定需要打印的loss名字， 这个名字需要跟loss的属性名字完全一致
        self.model_names = ['net',]  # 在这里指定网络模型的名字，这个对象的一个属性
        self.net = VGGPixel(in_channels=opt.in_channels, out_channel=3, channel_width=opt.channel_width, norm_type='batch')  # 模型的参数放在哪里呢

        # move net to GPU
        self.net.to(self.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, self.gpu_ids)


        self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()),
                                                lr=opt.lr)
        # 初始化损失函数, 检测的网络通常有很复杂的loss
        self.criterion = CentreLineLoss()

        # 会调用父类的setup方法为optimizers设置lr schedulers
        self.optimizers = []
        self.optimizers.append(self.optimizer)

    def set_input(self, input,):
        # dataset 中的数据会直接送到这里来，可以做一些简要的预处理. 大量的预处理应当放在dataset中做
        self.batch_images = torch.FloatTensor(input[0]).to(self.device)
        # self.label = input[1]  # 这个是一个list的
        self.heat_maps = torch.FloatTensor(input[1]).to(self.device)

    def forward(self):
        # 前向的写法， 可以写一些很复杂的前向途径, 比如多个模型组合，可以在这里自定义写一个前向传播
        self.score = self.net(self.batch_images)

    def backward(self):
        self.heatmap_loss, self.location_loss = self.criterion(self.heat_maps, self.score.permute(0, 2, 3, 1))
        self.loss = self.heatmap_loss + 0.1*self.location_loss
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()