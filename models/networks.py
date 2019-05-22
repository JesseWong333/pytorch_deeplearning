"""
networks中应当包含所有的主干网络的实现
包括
pixel-based model. 主要是有特征金字塔的一些融合操作
anchor-based model. 单层输出的比较简单
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
# from torchvision.models import VGG

# M: pooling为2的maxpooling, C pooling为1的maxpooling层， O 当层作为特征输出，传出这个索引
# 论文中说O应该在conv层之后，pooling层之前，但是看一个实现却是取pooling层之后
# 使用1/4的形式
cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'O',
          'M', 512, 512, 512, 'O', 'M', 512, 512, 512, 'O', 'C', 1024, 1024, 'O'],  # VGG_16 with extra layers
    'R': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
          'M', 512, 512, 512, 'M', 512, 512, 512],  # VGG_16 for RFB
}


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'switchable':
        from .switchable_norm import SwitchNorm2d
        norm_layer = functools.partial(SwitchNorm2d)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def make_layers(cfg, in_channels=3, channel_width = 1, norm_type='batch'):
    """

    :param cfg:
    :param in_channels: 1 or 3
    :param channel_width: 方便快速减少通道数量
    :param norm_type:
    :return:
    """
    norm_layer = get_norm_layer(norm_type)
    layers = []
    endpoint_index = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]  # 按照原pixel link中的写法
        elif v == 'O':
            endpoint_index.append(len(layers) - 1)  # 记住了是哪一层，但是没有记住通道数
        else:
            v = int(v*channel_width)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm_layer is not None:
                layers += [conv2d, norm_layer(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers, endpoint_index


def make_head(in_channel, out_channel):
    """

    :param in_channel: 一个list
    :param out_channel: 一个constant常量， 分开两个卷积和一个卷积实际上是没有区别的 (还是有区别的)
    :return:
    """
    head = []
    for i in in_channel:
        head += [nn.Conv2d(i, out_channel, kernel_size=1)]
    return head


# VGG 构成的pixel-based module, pixel link论文的结构
class VGGPixel(nn.Module):
    def __init__(self, in_channels=3, out_channel=6, channel_width=1, norm_type='batch'):
        super(VGGPixel, self).__init__()
        layers, endpoint_index = make_layers(cfg['D'], in_channels, channel_width, norm_type)
        self.vgg = nn.ModuleList(layers)
        endpoint_channel = [256, 512, 512, 1024]  # 就是cfg配置文件前面的数值
        self.head = nn.ModuleList(make_head(endpoint_channel, out_channel))
        self.score_head = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.endpoint_index = endpoint_index

    def forward(self, x):
        endpoint = []
        for k in range(len(self.vgg)):
            x = self.vgg[k](x)
            if k in self.endpoint_index:
                endpoint.append(x)

        # 对各个endpoint做一个卷积
        score_maps = []
        for x, head in zip(endpoint, self.head):
            score_maps.append(head(x))

        # 将各个score upsample, 然后再进行合并. 使用点加的方式
        # 最后两个feature map大小是相等的

        # 先写死吧
        feature = score_maps[2] + score_maps[3]
        for i in [1, 0]:
            feature = F.interpolate(feature, None, 2, 'bilinear') + score_maps[i]

        heat_map = self.score_head(feature)
        return heat_map


# ------------------------------------------------------------------------------
# 我需要单独的测试仅仅依靠anchor的RFB网络能够做到什么程度, 这里基于已有的VGG 写一个RFB
# RFB网络部分
# -------------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        '''
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


class VGGRFBNet(nn.Module):
    def __init__(self, in_channels=3, channel_width=1, norm_type='batch'):
        super(VGGRFBNet, self).__init__()
        layers, endpoint_index = make_layers(cfg['R'], in_channels, channel_width, norm_type)
        self.vgg = nn.ModuleList(layers)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        num_anchor = 7
        self.conf = nn.Sequential(*make_head([512], num_anchor*3))  # 3是类别数
        self.loc_head = nn.Sequential(*make_head([512], num_anchor*3))  # 输出头部的cx, cy, h
        self.loc_tail = nn.Sequential(*make_head([512], num_anchor*3))  # 输出尾部的3坐标 [xmax, ymin, ymax]

    def forward(self, x):
        for layer in self.vgg:
            x = layer(x)
        x = self.Norm(x)

        conf = self.conf(x).permute(0, 2, 3, 1).contiguous()  # channel放到最后
        loc_head = self.loc_head(x).permute(0, 2, 3, 1).contiguous()
        loc_tail = self.loc_tail(x).permute(0, 2, 3, 1).contiguous()

        # 拉值
        conf = conf.view(conf.size(0), -1)
        loc_head = loc_head.view(loc_head.size(0), -1)
        loc_tail = loc_tail.view(loc_tail.size(0), -1)

        # 拉成batch_size, num_prior, X
        conf = conf.view(conf.size(0), -1, 3)
        loc_head = loc_head.view(loc_head.size(0), -1, 3)
        loc_tail = loc_tail.view(loc_tail.size(0), -1, 3)

        return conf, loc_head, loc_tail


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


if __name__ == '__main__':
    # net = VGGPixel(in_channels=3, out_channel=1, channel_width=1, norm_type='batch')
    # d = net(torch.rand(1, 3, 256, 256))

    net = VGGRFBNet(in_channels=3, channel_width=1, norm_type='batch')
    d = net(torch.rand(1, 3, 256, 256))
    pass