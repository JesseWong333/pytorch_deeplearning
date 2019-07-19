# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/17

import torch.nn as nn

BACKBONE_REGISTRY = {}


def build_backbone(args):
    return BACKBONE_REGISTRY[args.backbone](args)


def register_backbone(cls):
    """Decorator to register a backbone, we do not need to register it with alias"""
    name = cls.__name__
    if name in BACKBONE_REGISTRY:
        raise ValueError('Cannot register duplicate backbone({})'.format(name))
    if not issubclass(cls, nn.Module):
        raise ValueError('Backbone ({}) must extend torch.nn.Module'.format(name))
    BACKBONE_REGISTRY[name] = cls
    return cls
