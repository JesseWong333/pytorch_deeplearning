# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/30

# hzc 添加对dataset的预处理操作的解耦, 分解成class, 然后再通过dict传入参数, 将预处理装载成pipeline
# todo: 怎样将pre_process和post_process 更好地处理, 每个模型都注册是不是不太好, 但是解耦有些麻烦, 有些后处理的代码一直在频繁地变动更新

data_root = '/media/Data/hzc/datasets/exam_number/frame/'
dataset_type = 'TextDetectionDataset'
voc_path = '/media/Data/hzc/datasets/exam_number/tokens.txt'

train_pipeline = [
    dict(type='LoadImageFromFile', gray=True),
    dict(type='AddNoise', with_salt=dict(prob=0.7), with_blur=dict(prob=0.7, min_val=1, max_val=5)),
    dict(type='RandomCrop', text_crop=dict(prob=0.5, padding=10)),
    dict(type='Resize', recog_scale=dict(imgH=32, factor=4)),
    dict(type='strLabelConverter', voc_path=voc_path),
]
test_pipeline = [
    dict(type='LoadImageFromFile', gray=True),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddNoise', with_salt=dict(prob=0.7), with_blur=dict(prob=0.7, min_val=1, max_val=5)),
    dict(type='RandomCrop', text_crop=dict(prob=0.5, padding=10)),
    dict(type='Resize', recog_scale=dict(imgH=32, factor=4)),
    dict(type='strLabelConverter', voc_path=voc_path),
]
dataset_type = 'TextRecognitionDataset'

config = dict(
    model='RecognitionModel',
    backbone=dict(
        type='TextRecognition',  # The name of the class
        num_class=11,
        FeatureExtraction=dict(
            type='RCNN',
            in_channels=1,
            out_channels=512,
            num_iteration=5),
        SequenceModeling=dict(
            type='None',
            layers=3,
            hidden_size=256,
            batch_first=True,
            dropout=0.2),
        Prediction=dict(
            type='CTC'),
        in_channels=3,
        out_channel=3,
        channel_width=1.0,
        norm_type='batch'),
    loss=dict(
        type='CTCLoss'
    ),
    # dataset = dict(
    #     type=dataset_type,
    #     pad_minw=144,
    #     collect_keys=('img', 'text', 'length'),
    #     ann_file=data_root + 'train_frame_1030.txt',
    #     isTrain=True,
    #     pipeline=train_pipeline,
    #
    # ),

    dataset=dict(
        train=dict(
            type=dataset_type,
            pad_minw=144,
            collect_keys=('img', 'text', 'length'),
            ann_file=data_root + 'train_frame_1030.txt',
            isTrain=True,
            pipeline=train_pipeline,
        ),
        val=dict(
            type=dataset_type,
            pad_minw=144,
            collect_keys=('img', 'text', 'length', 'filename', 'cpu_text'),
            ann_file=data_root + 'val_frame_1030.txt',
            isTrain=False,
            pipeline=test_pipeline,
        )
    ),
    require_evaluation=False, # 是否开启训练时评估的功能
    evaluator=dict(
        type="TextRecognitionEvaluator",
        vis_flag=True,
        vis_path='/media/Data/hzc/datasets/exam_number/frame_result',
    ),
    collate_fn=True,
    # 后处理分  todo: 后处理的默认参数设置。一个callable参数， 只给定部分参数
    post_process=dict(
        type='ctc_decoder',
        vocab_path=voc_path,
        vocab_type='col',
        subject='数学',
        beam_search=True
    ),
    #推理时的预处理
    pre_process=dict(
        type='recog_preprocess'
    ),

    isTrain=True,
    name='densenet_ctc',
    checkpoints_dir='./checkpoints',

    loss_ratios=None,
    loss_names=['ctc_loss'],  # the alias of the losses todo: is this a good way?

    batch_size=64,
    num_threads=4,
    gpu_ids=[3],

    # parameters of lr scheduler.
    lr=0.0005,
    lr_policy='linearly_decay',
    epoch=20, # 总的训练的epoch
    scheduler_param=[10, 10], # 前50个epoch适用lr, 再经过30 epoch 线性减为0

    init_type='xavier',

    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    load_models=[''],
    # load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based_TPS_OHEM_weighted/60_net_net.pth'],

    verbose=False,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5,
    eval_iter_freq=3000,
    eval_epoch_freq=5,
)