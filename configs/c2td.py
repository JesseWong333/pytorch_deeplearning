# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/30

config = dict(
    model='VanillaModel',
    backbone=dict(
        type='VGGPixel',  # The name of the class
        in_channels=3,
        out_channel=3,
        channel_width=1.0,
        norm_type='batch'
    ),
    loss=dict(
        type='CentreLineLoss',
        thresh=0.5,
        neg_pos=3
    ),
    dataset=dict(
        type='ImageLine2Dataset',
        dataroot='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
    ),

    isTrain=True,
    name='c2td',
    checkpoints_dir='./checkpoints',


    loss_ratios=[1, 0.1],
    loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses

    batch_size=2,
    num_threads=5,
    gpu_ids=[1],

    # parameters of lr scheduler.
    lr=0.001,
    lr_policy='linearly_decay',
    epoch=80,
    scheduler_param=[50, 30],  # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    continue_model='',

    verbose=False,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5
)