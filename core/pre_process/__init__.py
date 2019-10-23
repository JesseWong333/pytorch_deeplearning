import os
import importlib
import types

"""
对于推理前的预处理应该能够注册两种类型的结构，
只要是注册的参数就应该单独使用一个dict来注册
1） 有参数的类类型，自定义的class.  应当在cfg中使用type来包括一个类型， 其余的使用
2） 有参数的函数类型
参考工程里其他的注册改的  by hcn
"""

PRE_PROCESS_REGISTRY = {}


def build_pre_process(args):
    obj_type = args.pop('type')
    '''
    https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
    Use 'callable' to detect whether a Python variable is a function, not isinstance(x, function)
    But a class is also callable
    '''
    if isinstance(PRE_PROCESS_REGISTRY[obj_type], types.FunctionType):
        return PRE_PROCESS_REGISTRY[obj_type]
    else:
        return PRE_PROCESS_REGISTRY[obj_type](**args)  # initialize the post process object

def register_pre_process(cls):
    name = cls.__name__
    if name in PRE_PROCESS_REGISTRY:
        raise ValueError('Cannot register duplicate post_process({})'.format(name))
    PRE_PROCESS_REGISTRY[name] = cls
    return cls

# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.pre_process.' + module)