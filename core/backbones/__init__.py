# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/17
import os
import importlib
import torch.nn as nn

BACKBONE_REGISTRY = {}


def build_backbone(args):
    obj_type = args.pop('type')  # 参照mmdetection的config做法，使用TYPE作为主key
    return BACKBONE_REGISTRY[obj_type](**args)


def register_backbone(cls):
    """Decorator to register a backbone, we do not need to register it with alias"""
    name = cls.__name__
    if name in BACKBONE_REGISTRY:
        raise ValueError('Cannot register duplicate backbone({})'.format(name))
    if not issubclass(cls, nn.Module):
        raise ValueError('Backbone ({}) must extend torch.nn.Module'.format(name))
    BACKBONE_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.backbones.' + module)
