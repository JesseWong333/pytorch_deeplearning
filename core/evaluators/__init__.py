# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/17
"""
add by hzc
add evaluators for eavluation
"""

import os
import importlib
import torch.nn as nn

EVALUATORS_REGISTRY = {}


def build_evaluator(args):
    obj_type = args.evaluator.pop('type')  # 参照mmdetection的config做法，使用TYPE作为主key
    return EVALUATORS_REGISTRY[obj_type](**args.evaluator)


def register_evaluator(cls):
    """Decorator to register a backbone, we do not need to register it with alias"""
    name = cls.__name__
    if name in EVALUATORS_REGISTRY:
        raise ValueError('Cannot register duplicate evaluator({})'.format(name))
    # if not issubclass(cls, nn.Module):
    #     raise ValueError('Evaluator ({}) must extend torch.nn.Module'.format(name))
    EVALUATORS_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.evaluators.' + module)
