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
    # loss=dict(
    #     type='CentreLineLoss',
    #     thresh=0.5,
    #     neg_pos=3
    # ),
    # dataset=dict(
    #     type='ImageLine2Dataset',
    #     dataroot='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
    # ),
    # 后处理分  todo: 后处理的默认参数设置。一个callable参数， 只给定部分参数
    post_process=dict(
        type='pixel_link_process'
    ),

    isTrain=True,
    name='pixel_link',
    checkpoints_dir='./checkpoints',


    # loss_ratios=[1, 0.1],
    # loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses todo: is this a good way?
    #
    # batch_size=2,
    # num_threads=5,
    gpu_ids=[3],

    # parameters of lr scheduler.
    lr=0.001,
    lr_policy='linearly_decay',
    epoch=80,  # 总的训练的epoch
    scheduler_param=[50, 30],  # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    init_type='xavier',


    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    load_models=['/media/Data/hcn/project/pytorch-ocr-framework/checkpoints/pixel_link_0806/epoch_99_pixellinkmodel.pth'],
    # load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based_TPS_OHEM_weighted/60_net_net.pth'],

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