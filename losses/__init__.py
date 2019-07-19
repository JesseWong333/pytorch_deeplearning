import torch.nn as nn

LOSS_REGISTRY = {}


def build_loss(args):
    if LOSS_REGISTRY.get(args.loss) is None:
        return None  # todo: import non custom loss
    return LOSS_REGISTRY[args.loss](args)


def register_backbone(cls):
    name = cls.__name__
    if name in LOSS_REGISTRY:
        raise ValueError('Cannot register duplicate backbone({})'.format(name))
    if not issubclass(cls, nn.Module):
        raise ValueError('Backbone ({}) must extend torch.nn.Module'.format(name))
    LOSS_REGISTRY[name] = cls
    return cls
