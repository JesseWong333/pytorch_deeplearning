# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/20

config = dict(
    model='VanillaModel',
    backbone=dict(
        type='Unet',  # The name of the class
        input_nc=3,
        output_nc=5,
        num_downs=5
    ),
    loss=dict(
        type='SegmentationOHEMLoss',
        neg_pos=3
    ),
    dataset=dict(
        train=dict(
        type='SteelDefectDataset',
        csv_file='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/severstal-steel-defect-detection/train.csv',
        folder='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/severstal-steel-defect-detection/train_images'),
    val=dict()
    ),
    require_evaluation=False,
    evaluator=dict(
        type="TextRecognitionEvaluator",
        vis_flag=True,
        vis_path='/media/Data/hzc/datasets/exam_number/frame_result',
    ),
    collate_fn=True,

    # 后处理分  todo: 后处理的默认参数设置。一个callable参数， 只给定部分参数
    post_process=dict(
        type='convert_seg_to_encoded_pixels'
    ),

    isTrain=True,
    name='steel_defect_unet',
    checkpoints_dir='./checkpoints',

    # loss没有组合
    # loss_ratios=[1, 0.1],
    # loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses todo: is this a good way?

    batch_size=36,
    num_threads=12,
    gpu_ids=[1, 2, 3],

    # parameters of lr scheduler.
    lr=0.001,
    lr_policy='linearly_decay',
    epoch=80,  # 总的训练的epoch
    scheduler_param=[50, 30],  # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    init_type='xavier',


    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    # load_models=5,
    load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/steel_defect_unet/5_net_net.pth'],

    verbose=False,
    print_freq=30,
    save_latest_freq=5000,
    save_epoch_freq=5,
    eval_iter_freq=6000,
    eval_epoch_freq=5,
)