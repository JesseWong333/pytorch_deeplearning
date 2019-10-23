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
        thresh=0.5,
        neg_pos=3
    ),
    dataset=dict(
        type='ImageIm2LatexDataset',
        dataroot='/home/chen/hcn/data/formula/formula_images/labels/0917/'
    ),
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
    checkpoints_dir='/media/Data/hcn/project/pytorch-ocr-framework/checkpoints',


    # loss_ratios=[1, 0.1],
    # loss_names=['heatmap_loss', 'location_loss'],  # the alias of the losses todo: is this a good way?
    #
    batch_size=6,
    num_threads=5,
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
    load_models=['../model_files/epoch_11_im2latexmodel.pth'],  # 支持epoch导入，或者直接pth导入
    # load_models=['/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based_TPS_OHEM_weighted/60_net_net.pth'],

    verbose=True,
    print_freq=100,
    save_latest_freq=5000,
    save_epoch_freq=5,

    im2latex_congigs='../configs/im2latex.yaml'
)