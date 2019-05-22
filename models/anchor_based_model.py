import torch
import itertools
from .networks import VGGRFBNet
from .base_model import BaseModel
from losses.multibox_loss import MultiBoxHeadLoss, MultiBoxTailLoss
from losses.prior_box_eval import PriorBoxEval


cfg = {
    'feature_maps' : [(32, 64)],
    'min_dim' : (512, 1024),
    'steps' : [16],
    # 在15~70之间均匀
    'defaultbox': [
        [16, 16],
        [27, 27],
        [30, 30],
        [40, 40],
        [50, 50],
        [60, 60],
        [70, 70]],
    'variance' : [0.1, 0.2],
    'clip' : False,
}


class AnchorBasedModel(BaseModel):
    def name(self):
        return "AnchorBasedModel"

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
        # 在这里指定需要打印的loss名字， 这个名字需要跟loss的属性名字完全一致
        self.loss_names = ['loss', 'loss_head_l', 'loss_head_c', 'loss_tail_l', 'loss_tail_c']
        self.model_names = ['net',]  # 在这里指定网络模型的名字，这个对象的一个属性
        self.net = VGGRFBNet(in_channels=opt.in_channels, channel_width=opt.channel_width, norm_type='batch')

        # move net to GPU
        self.net.to(self.gpu_ids[0])
        self.net = torch.nn.DataParallel(self.net, self.gpu_ids)
        self.optimizer = torch.optim.Adam(itertools.chain(self.net.parameters()), lr=opt.lr)

        # 在网络中，我将头尾两类分别处理，一个位置可以同时为头和尾。
        self.criterion_head = MultiBoxHeadLoss(num_classes=2, overlap_thresh=0.5, neg_pos=3)
        self.criterion_tail = MultiBoxTailLoss(num_classes=2, overlap_thresh=0.5, neg_pos=3)

        # 会调用父类的setup方法为optimizers设置lr schedulers
        self.optimizers = []
        self.optimizers.append(self.optimizer)

        # 生成基于anchor的priors
        priorbox = PriorBoxEval(cfg)
        self.priors = priorbox.forward()

    def set_input(self, input,):
        # dataset 中的数据会直接送到这里来，可以做一些简要的预处理. 大量的预处理应当放在dataset中做
        self.batch_images = torch.FloatTensor(input[0]).to(self.device)
        height = self.batch_images.shape[2]
        width = self.batch_images.shape[3]
        batch_boxes = input[1]  # 这个是一个list的
        # convert boxes into a list of tensor, 并将坐标变为相对坐标
        self.targets = []
        for boxes in batch_boxes:
            boxes[:, [0, 2]] /= float(width)  # 这个相对好不好？ 长和宽部分
            boxes[:, [1, 3]] /= float(height)
            self.targets.append(torch.from_numpy(boxes).to(self.device))

        self.heat_maps = torch.FloatTensor(input[2]).unsqueeze(1).to(self.device)

    def forward(self):
        # 前向的写法， 可以写一些很复杂的前向途径, 比如多个模型组合，可以在这里自定义写一个前向传播
        self.net_out = self.net(self.batch_images)

    def backward(self):
        self.loss_head_l, self.loss_head_c = self.criterion_head(self.net_out, self.priors, self.targets)  # 生成前项的prior
        self.loss_tail_l, self.loss_tail_c = self.criterion_tail(self.net_out, self.priors, self.targets)
        self.loss = self.loss_head_l + self.loss_head_c + self.loss_tail_l + self.loss_tail_c
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()