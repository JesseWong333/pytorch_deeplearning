import os
import importlib
import torch.nn as nn

LOSS_REGISTRY = {}


def build_loss(args):
    obj_type = args.pop('type')
    if LOSS_REGISTRY.get(obj_type) is None:
        return None  # todo: import non custom loss
    return LOSS_REGISTRY[obj_type](**args)


def register_loss(cls):
    name = cls.__name__
    if name in LOSS_REGISTRY:
        raise ValueError('Cannot register duplicate loss({})'.format(name))
    if not issubclass(cls, nn.Module):
        raise ValueError('loss ({}) must extend torch.nn.Module'.format(name))  # 自定义的loss需要继承nn.Module吗？
    LOSS_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.losses.' + module)
