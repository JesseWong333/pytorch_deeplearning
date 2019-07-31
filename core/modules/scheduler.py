# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/31
from torch.optim import lr_scheduler


def get_scheduler(optimizer, args):
    if args.lr_policy == 'linearly_decay':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + args.epoch_count - args.scheduler_param[0]) / float(args.scheduler_param[1] + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # 自己写的lambda rule
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler