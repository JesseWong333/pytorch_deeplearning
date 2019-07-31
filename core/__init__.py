from .base_model import BaseModel
import importlib
import os

MODEL_REGISTRY = {}


def build_model(args):
    obj_type = args.model
    instance = MODEL_REGISTRY[obj_type]()
    instance.initialize(args)
    print("model [%s] was created" % obj_type)
    return instance


def register_model(cls):
    name = cls.__name__
    if name in MODEL_REGISTRY:
        raise ValueError('Cannot register duplicate model({})'.format(name))
    if not issubclass(cls, BaseModel):
        raise ValueError('model ({}) must extend torch.BaseModel'.format(name))
    MODEL_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.' + module)
