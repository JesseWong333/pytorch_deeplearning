import argparse
import os
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # 实验的基本配置：名字，使用的模型, 训阶段
        parser.add_argument('--name', type=str, default='skew_model',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--model', type=str, default='skew',
                            help='chooses which model to use.')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # 杂项
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # 数据相关： 数据路径、使用的datasets、装载数据的线程数、batch_sizes
        parser.add_argument('--dataroot', type=str, default='/media/Data/wangjunjie_code/pytorch_text_detection/datasets/text_images/',
                            help='path to images')
        parser.add_argument('--dataset', type=str, default='tps_dataset', help='which customized dataset to use?')
        parser.add_argument('--num_threads', default=5, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', default=5, type=int, help='batch size')

        # 模型存储相关： 存储路径，多少epoch存储一次. 暂存为latest模型的时间
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')  # 多长时间存储一次，以步来计算

        # 模型重新装载的参数
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')


        # epoch, 显示频率等参数
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')

        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')


        # =========================================================================== #
        # lr policy 在neteorks.py get_scheduler中定义
        # 一共定义了5种不同的lr scheduler.
        # 取决于先选择了哪一样lr调节策略， 相应的策略参数才会起作用
        # =========================================================================== #
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau|cosine')

        # =====================================================================
        # lr_scheduler.LambdaLR
        # =====================================================================
        # niter 表示从这个epoch开始调节学习率， niter_decay 表示再经过这个多个learning rate 调节为0
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')

        # =====================================================================
        # lr_scheduler.StepLR
        # =====================================================================
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        if opt.phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
