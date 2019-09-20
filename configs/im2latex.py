# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/20

config = dict(
    model='Im2LatexModel',
    # backbone=dict(
    #     type='VGGPixelWithDilation',  # The name of the class
    #     in_channels=3,
    #     out_channel=10,
    #     channel_width=1.0,
    #     norm_type='batch'
    # ),
    # loss=dict(
    #     type='CentreLineLoss',
    #     thresh=0.5,
    #     neg_pos=3
    # ),
    # dataset=dict(
    #     type='ImageLine2Dataset',
    #     dataroot='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
    # ),
    # 后处理分  todo: 后处理也要调用全局的args呢，写在前面？
    post_process=dict(
        type='Im2latex_postprocess',
        n_best=1,
        replace_unk=False,
        vocab_cfg=dict(
            unk="_UNK",
            pad="_PAD",
            end="_END",
            init="_INIT",
            path_vocab="/media/Data/hzc/datasets/formulas/0603/tokens-utf-8.txt",
            min_count_tok=2
        )
    ),

    isTrain=True,
    name='im2latex',
    checkpoints_dir='/media/Data/hcn/project/pytorch-ocr-framework/checkpoints',


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
    load_models=['/media/Data/hcn/project/pytorch-ocr-framework/checkpoints/im2latex/epoch_10_im2latexmodel.pth'],  # 支持epoch导入，或者直接pth导入
    # load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based_TPS_OHEM_weighted/60_net_net.pth'],

    verbose=True,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5,

    im2latex_congigs='../configs/im2latex.yaml'
)