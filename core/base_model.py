import os
import torch
from collections import OrderedDict
from .modules import get_scheduler
from .modules import init_weights


class BaseModel():

    def name(self):
        return 'BaseModel'

    def initialize(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
        self.loss_names = []
        self.model_names = []
        self.gpu_ids = args.gpu_ids
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)
        if args.isTrain:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.output_names = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, args):
        if args.isTrain:
            self.schedulers = [get_scheduler(optimizer, args) for optimizer in self.optimizers]
        self.print_networks(args.verbose)
        if not args.isTrain or args.continue_train:
            if isinstance(args.load_models, int):  # 如果是一个数值，就寻找原路径按照epoch
                self.load_networks_epoch(args.load_models)
            if isinstance(args.load_models, list):  # 如果是一个路径list(可能包含多个模型)，就按照实际去load
                self.load_neteorks_list(args.load_models)
        else:
            # init network
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, name)
                    init_weights(net, args.init_type)

    # make core eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        visual_ret = []
        for name in self.output_names:
            if isinstance(name, str):
                visual_ret.append(getattr(self, name))
        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                values = getattr(self, name)
                if isinstance(values, tuple) or isinstance(values, list):
                    for error, loss_name in zip(values, self.args.loss_names):
                        errors_ret[loss_name] = float(error)
                else:
                    errors_ret[name] = float(getattr(self, name))
        return errors_ret

    # 原来是将打印的信息放到专门的visualizer， 这里集成到了model中
    def print_train_info(self, epoch, i, t, t_data):
        losses = self.get_current_losses()
        #
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        # with open(self.log_name, "a") as log_file:
        #     log_file.write('%s\n' % message)

    # save core to the disk
    def save_networks(self, epoch):
        for name in self.model_names:
            print(name)
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # net.cuda(self.gpu_ids[0])#remove by hcn. Why to do this operation???
                else:
                    torch.save(net.state_dict(), save_path)

    def load_networks(self, name, load_path):
        net = getattr(self, name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        # if hasattr(state_dict, '_metadata'):
        #     del state_dict._metadata
        #
        # # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(self.remove_moudle(state_dict))

    @staticmethod
    def remove_moudle(state_dict):
        """本例子中的在存储网络的时候已经全部去掉了module开始的，为了兼容其他的，加上判断"""
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            # print(name)
            new_state_dict[name] = v
        return new_state_dict

    # 去掉了Instance Norm中的参数，有现在已经没有这个必要了吧?
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_neteorks_list(self, paths):
        for name, path in zip(self.model_names, paths):
            self.load_networks(name, path)

    def load_networks_epoch(self, epoch):
        for name in self.model_names:
            load_filename = '%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(self.save_dir, load_filename)
            self.load_networks(name, load_path)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
