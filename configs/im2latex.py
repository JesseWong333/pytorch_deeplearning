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
    loss=dict(
        type='Im2TextLoss',
        ignore_index=1,#padding_id
        reduction='sum',
        lambda_coverage=0
    ),
    dataset=dict(
        train=dict(
            type='ImageIm2LatexDataset',
            dataroot='/home/chen/hcn/data/formula/formula_images/labels/0917/'),
        val=dict(
        )
    ),
    require_evaluation=False,
    evaluator=dict(
        type="TextRecognitionEvaluator",
        vis_flag=True,
        vis_path='/media/Data/hzc/datasets/exam_number/frame_result',
    ),
    collate_fn=True,

    # 后处理分  todo: 后处理也要调用全局的args呢，写在前面？
    Vocab=dict(
        unk="_UNK",
        pad="_PAD",
        end="_END",
        init="_INIT",
        path_vocab="/home/chen/hcn/data/formula/formula_images/labels/0917/tokens-utf-8.txt",
        min_count_tok=2
    ),
    Model=dict(
        encoder_cnn="vanilla",
        positional_embeddings=True,
        attn_cell_config=dict(
            cell_type="lstm",
            num_units=512,
            dim_e=512,
            dim_o=512,
            dim_embeddings=80
        ),
        decoding="beam_search",
        beam_size=5,
        div_gamma=1,
        div_prob=0,
        max_length_formula=150,
        emb_dim=80,
        enc_rnn_h=256,
        dec_rnn_h=512,
        clip_grad=True,
        max_grad_norm=20
    ),
    post_process=dict(
        type='Im2latex_postprocess',
        n_best=1,
        replace_unk=False,
        vocab_cfg=dict(
            unk="_UNK",
            pad="_PAD",
            end="_END",
            init="_INIT",
            path_vocab="../model_files/tokens-utf-8.txt",
            min_count_tok=2
        )
    ),
    pre_process=dict(
        type='im2latex_preprocess'
    ),

    isTrain=True,
    name='im2latex',
    checkpoints_dir='/home/chen/hcn/project/pytorch_deeplearning/checkpoints',


    # loss_ratios=[1, 0.1],
    # loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses todo: is this a good way?
    #
    batch_size=6,
    num_threads=5,
    gpu_ids=[3],

    # parameters of lr scheduler.
    lr=0.0001,
    lr_policy='linearly_decay',
    epoch=16,  # 总的训练的epoch
    scheduler_param=[8, 8],  # 前8个epoch适用lr, 再经过8 epoch 线性减为0

    init_type='xavier',

    # parameters of continuing to train the model
    epoch_count=1,  # 如果是重新开始训练，该值始终应该为1
    continue_train=False,
    load_models=['/home/chen/hcn/project/pytorch_deeplearning/checkpoints/im2latex/14_net_net.pth'],  # 支持epoch导入，或者直接pth导入
    # load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based_TPS_OHEM_weighted/60_net_net.pth'],

    verbose=True,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=1,
    eval_iter_freq=3000,
    eval_epoch_freq=5,
)