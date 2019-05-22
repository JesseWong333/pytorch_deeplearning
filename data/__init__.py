#  要能够在__init__ 文件中直接指定用哪一个data set,  返回的应该是一个data loader

import importlib
import torch


def find_dataset_using_name(dataset_name):
    # 要求文件的名字和需要导入的库的名字应该一致
    model_filename = "data." + dataset_name  # py文件的名字
    modellib = importlib.import_module(model_filename)

    model = None

    target_model_name = dataset_name.replace('_', '')  # 要导入的文件的名字
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should a dataset named %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_data_loader(opt):
    model = find_dataset_using_name(opt.dataset)
    instance = model(opt)
    print("using [%s]" % (instance.name()))
    data_loader = torch.utils.data.DataLoader(
        instance, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_threads,
        collate_fn=instance.collate_fn)
    return data_loader
