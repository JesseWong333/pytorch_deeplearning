# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/6

from torch.nn import init


def init_weights(net, init_type):
    """
    from pytorch cycle-GAN
    这种针对BN层和bias单独设置的做法是不是一种标准的固定做法?
    :param net:
    :param init_type:
    :return:
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data,)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, )
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    # Applies fn recursively to every submodule
    net.apply(init_func)