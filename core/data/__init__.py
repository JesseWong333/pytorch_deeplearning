import torch.utils.data as data
import importlib
import os
from torch.utils.data import _utils

default_collate = _utils.collate.default_collate

DATASET_REGISTRY = {}


def build_dataset(args, isTrain=True):
    dataset_cfg = args.dataset.train if isTrain else args.dataset.test
    obj_type = dataset_cfg.pop('type')
    # 创建dataloader 返回
    instance = DATASET_REGISTRY[obj_type](**dataset_cfg)
    data_loader = data.DataLoader(
        instance, shuffle=True, batch_size=args.batch_size, num_workers=args.num_threads,
        collate_fn=instance.collate_fn if args.collate_fn else default_collate)
    return data_loader


def register_dataset(cls):
    name = cls.__name__
    if name in DATASET_REGISTRY:
        raise ValueError('Cannot register duplicate Dataset({})'.format(name))
    if not issubclass(cls, data.Dataset):
        raise ValueError('loss ({}) must extend data.Dataset'.format(name))
    DATASET_REGISTRY[name] = cls
    return cls


# automatically import any Python files in the directory. while import the python file, register methods called
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('core.data.' + module)
