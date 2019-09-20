# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/17
# @ author: Jesse Wang

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import get_norm_layer, BasicRFB_a
from . import register_backbone
from torch.optim import lr_scheduler
import functools


"""
包含所有需要VGG的模型. VGG的FPN. 如果有需要根据可根据已有的make_layers进行等函数创建
"""

# M: pooling为2的maxpooling, C pooling为1的maxpooling层， O 当层作为特征输出，传出这个索引
# 论文中说O应该在conv层之后，pooling层之前，但是看一个实现却是取pooling层之后
# 使用1/4的形式
# M: pooling 层
# O: endpoint
# C: 自定义的层
# 注意，在实现的过程中尽量保证兼容

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'O',
          'M', 512, 512, 512, 'O', 'M', 512, 512, 512, 'O', 'C', 1024, 1024, 'O'],  # VGG_feature_pyramid
    'P': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'O',
          'M', 512, 512, 512, 'O', 'M', 512, 512, 512, 'O', 'C', 'C', 'C', 'O'],  # VGG_feature_pyramid_Dilation
    'R': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
          'M', 512, 512, 512, 'M', 512, 512, 512],  # VGG_16 for RFB
    'S': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M', 512, 'M', 512, ],  # VGG with extra layers. 需要有6个pooling层

}


def make_layers(cfg, in_channels=3, channel_width = 1, norm_type='batch', custom_layer=None):
    """

    :param cfg:
    :param in_channels: 1 or 3
    :param channel_width: 方便快速减少通道数量
    :param norm_type:
    :param custom_layer: 自定义的层， 以C表示
    :return:
    """
    norm_layer = get_norm_layer(norm_type)
    layers = []
    endpoint_index = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += custom_layer.pop(0)  # 按照原pixel link中的写法
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
@register_backbone
class VGGPixel(nn.Module):
    def __init__(self, in_channels=3, out_channel=6, channel_width=1, norm_type='batch'):
        super(VGGPixel, self).__init__()
        c_layers = [[nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]]
        layers, endpoint_index = make_layers(cfg['D'], in_channels, channel_width, norm_type, c_layers)
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

        feature = score_maps[2] + score_maps[3]
        for i in [1, 0]:
            feature = F.interpolate(feature, None, 2, 'bilinear') + score_maps[i]
        heat_map = self.score_head(feature)
        return heat_map


# VGG Pixel-link给出的结构，在一层有Dilation
@register_backbone
class VGGPixelWithDilation(nn.Module):
    def __init__(self, in_channels=3, out_channel=6, channel_width=1, norm_type='batch'):
        super(VGGPixelWithDilation, self).__init__()
        c_layers = [[nn.MaxPool2d(kernel_size=3, stride=1, padding=1)],
                    [nn.Conv2d(int(512*channel_width), int(1024*channel_width), kernel_size=3, padding=6, dilation=6),
                     get_norm_layer(norm_type)(int(1024*channel_width)),
                     nn.ReLU(inplace=True)
                     ],
                    [nn.Conv2d(int(1024*channel_width), int(1024*channel_width), kernel_size=3, padding=1),
                     get_norm_layer(norm_type)(int(1024 * channel_width)),
                     nn.ReLU(inplace=True)]
                    ]
        layers, endpoint_index = make_layers(cfg['P'], in_channels, channel_width, norm_type, c_layers)
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

        score_maps = []
        for x, head in zip(endpoint, self.head):
            score_maps.append(head(x))

        feature = score_maps[2] + score_maps[3]
        for i in [1, 0]:
            feature = F.interpolate(feature, None, 2, 'bilinear') + score_maps[i]
        heat_map = self.score_head(feature)
        return heat_map


@register_backbone
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