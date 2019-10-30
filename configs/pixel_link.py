# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/19

config = dict(
    model='VanillaModel',
    backbone=dict(
        type='VGGPixelWithDilation',  # The name of the class
        in_channels=3,
        out_channel=10,
        channel_width=1.0,
        norm_type='batch'
    ),
    loss=dict(
        type='PixelLinkLoss'
    ),
    dataset=dict(
        type='ImagePixelLinkDataset',
        dataroot='/home/chen/hcn/data/pixel_link_data/0517/train/',
        config=dict(
            score_map_shape=(128, 256),
            stride=4,
            pixel_cls_weight_method="PIXEL_CLS_WEIGHT_bbox_balanced",
            text_label=1,
            ignore_label=-1,
            background_label=0,
            num_neighbours=4,
            bbox_border_width=1,
            pixel_cls_border_weight_lambda=1.0,
            pixel_neighbour_type="PIXEL_NEIGHBOUR_TYPE_4",
            decode_method="DECODE_METHOD_join",
            min_area=10,
            min_height=1,
            pixel_conf_threshold=0.7,
            link_conf_threshold=0.5,
            pixel_mean=[0.91610104, 0.91612387, 0.91603917],
            pixel_std=[0.2469206, 0.24691878, 0.24692364]
        )
    ),
    # 后处理分  todo: 后处理的默认参数设置。一个callable参数， 只给定部分参数
    post_process=dict(
        type='pixel_link_process'
    ),
    pre_process=dict(
        type='pixellink_preprocess'
    ),

    isTrain=True,
    name='pixel_link',
    checkpoints_dir='./checkpoints',


    loss_ratios=[1, 1],
    loss_names=['cls_loss', 'link_loss'],  # the alias of the losses todo: is this a good way?
    #
    batch_size=2,
    num_threads=5,
    gpu_ids=[2],

    # parameters of lr scheduler.
    lr=0.001,
    lr_policy='linearly_decay',
    epoch=150,  # 总的训练的epoch
    scheduler_param=[80, 70],  # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    init_type='xavier',


    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    # load_models=['../model_files/epoch_99_pixellinkmodel.pth'],
    load_models=['/home/chen/hcn/project/pytorch_deeplearning/checkpoints/pixel_link/latest_net_net.pth'],

    verbose=True,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5,

    # pixle_link 在多个地方用到的配置
    pixel_link=dict(
        score_map_shape=(128, 256),
        stride=4,
        pixel_cls_weight_method="PIXEL_CLS_WEIGHT_bbox_balanced",
        text_label=1,
        ignore_label=-1,
        background_label=0,
        num_neighbours=4,
        bbox_border_width=1,
        pixel_cls_border_weight_lambda=1.0,
        pixel_neighbour_type="PIXEL_NEIGHBOUR_TYPE_4",
        decode_method="DECODE_METHOD_join",
        min_area=10,
        min_height=1,
        pixel_conf_threshold=0.7,
        link_conf_threshold=0.5,
    )
)