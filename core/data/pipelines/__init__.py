"""
add by hzc
将常用的预处理代码解耦成pipeline, 供配置文件传dict参数调用
"""

import os
import importlib
import types
PIPELINE_REGISTRY = {}


def build_pipeline(args):
    obj_type = args.pop('type')
    '''
    https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
    Use 'callable' to detect whether a Python variable is a function, not isinstance(x, function)
    But a class is also callable
    '''
    if isinstance(PIPELINE_REGISTRY[obj_type], types.FunctionType):
        return PIPELINE_REGISTRY[obj_type]
    else:
        return PIPELINE_REGISTRY[obj_type](**args)  # initialize the post process object


def register_pipeline(cls):
    name = cls.__name__
    if name in PIPELINE_REGISTRY:
        raise ValueError('Cannot register duplicate post_process({})'.format(name))
    PIPELINE_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    # print("file", file)
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        # print(module)
        importlib.import_module('core.data.pipelines.' + module)