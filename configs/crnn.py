# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/1


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
        type='CRNN',
        imgH=32,
        nc=1,
        nclass=6725,  # 具体的类别数通过代码中获取字典再设置
    ),
    # loss=dict(
    #     type='CentreLineLoss',
    #     thresh=0.5,
    #     neg_pos=3
    # ),
    # dataset=dict(
    #     train=dict(
    #     type='ImageLine2Dataset',
    #     dataroot='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
    #     ),
    #     val=dict()
    # ),

    require_evaluation=False,
    evaluator=dict(
        type="TextRecognitionEvaluator",
        vis_flag=True,
        vis_path='/media/Data/hzc/datasets/exam_number/frame_result',
    ),
    collate_fn=True,

    # 后处理分  todo: 后处理的默认参数设置。一个callable参数， 只给定部分参数
    post_process=dict(
        type='ctc_decoder',
        convert_list_path='/home/chen/hcn/project/pytorch_deeplearning/files/char_std_print_6725.txt'
    ),

    isTrain=True,

    name='crnn',
    checkpoints_dir='./checkpoints',


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
    load_models=['/home/chen/hcn/project/pytorch_deeplearning/files/netCRNN_1_66000_2-1_new.pth'],

    verbose=False,
    # print_freq=100,
    # save_latest_freq=5000,
    # save_epoch_freq=5
)