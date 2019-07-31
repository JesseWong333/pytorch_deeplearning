import torch.utils.data as data
import importlib
import os

DATASET_REGISTRY = {}


def build_dataset(args):
    obj_type = args.dataset.pop('type')
    # 创建dataloader 返回
    instance = DATASET_REGISTRY[obj_type](**args.dataset)
    data_loader = data.DataLoader(
        instance, shuffle=True, batch_size=args.batch_size, num_workers=args.num_threads,
        collate_fn=instance.collate_fn)
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
