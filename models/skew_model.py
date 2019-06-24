import torch
import itertools
from .networks import VGGSkew
from .base_model import BaseModel
from torchvision.transforms.functional import to_tensor


class SkewModel(BaseModel):
    def name(self):
        return "SkewModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # called at model.__init__.py then at base_option.py gather_options
        # 在这里写当前模型的参数
        parser.add_argument('--in_channels', default=3, type=int,
                            help='the in_channel of the image,  should be 1 or 3')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # 在这里初始化模型, 初始化optimizer,
        # 网络的train() 和 eval() 模式可以在这里切换
        self.loss_names = ['loss',]  # 在这里指定需要打印的loss名字， 这个名字需要跟loss的属性名字完全一致
        self.model_names = ['net',]  # 在这里指定网络模型的名字，这个对象的一个属性
        self.net = VGGSkew(in_channels=3, channel_width=1, norm_type='batch')

        # move net to GPU
        self.net.to(self.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, self.gpu_ids)

        self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()),
                                                lr=opt.lr)
        # 初始化损失函数, 检测的网络通常有很复杂的loss
        self.criterion = torch.nn.L1Loss()

        # 会调用父类的setup方法为optimizers设置lr schedulers
        self.optimizers = []
        self.optimizers.append(self.optimizer)

    def set_input(self, input,):
        # dataset 中的数据会直接送到这里来，可以做一些简要的预处理. 大量的预处理应当放在dataset中做
        self.warped_img = input[0].to(self.device)
        self.target = input[1].to(self.device)

    def forward(self):
        # 前向的写法， 可以写一些很复杂的前向途径, 比如多个模型组合，可以在这里自定义写一个前向传播
        self.predict = self.net(self.warped_img)

    def backward(self):
        self.loss = self.criterion(self.predict, self.target)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()