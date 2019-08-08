# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/6


'''
在infer 阶段不需要的
1) 生成optimizer 和 loss. 在eval阶段是需要生成loss的， 不过这个阶段可以理解为在train
   以及跟优化器相关的参数：
   lr scheduler
2) 创建dataset
2) 不需要字段： name, checkpoints_dir, loss_ratios, loss_names, batch_size, num_threads
请仔细检查在 test 阶段中， 是否还有出现这些字段的引用
'''


config = dict(
    model='VanillaModel',
    backbone=dict(
        type='GRCNN',
        imgH=32,
        nc=1,
        nclass=6595,  # 具体的类别数通过代码中获取字典再设置
        nh=256
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
        type='ctc_decoder',
        convert_list_path='/media/Data/wangjunjie_code/pytorch_text_detection/files/char_std_xunfei_6595.txt'
    ),

    isTrain=True,

    # name='c2td',
    # checkpoints_dir='./checkpoints',


    # loss_ratios=[1, 0.1],
    # loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses todo: is this a good way?

    # batch_size=2,
    # num_threads=5,

    gpu_ids=[2],

    # parameters of lr scheduler.
    # lr=0.001,
    # lr_policy='linearly_decay',
    # epoch=80,  # 总的训练的epoch
    # scheduler_param=[50, 30],  # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    # parameters of continuing to train the model
    # epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    # continue_train=False,
    load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/files/netGRCNN_6_7000.pth'],

    verbose=False,
    # print_freq=100,
    # save_latest_freq=5000,
    # save_epoch_freq=5
)