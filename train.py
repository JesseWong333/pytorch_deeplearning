import time
from visdom_visualize import Visualizer
from core.data import build_dataset  # 这个位置就已经导入了，怎么办 在import的时候代码执行
from core import build_model
from configs.c2td import config  # todo argparse指定配置文件
from core.evaluators import build_evaluator
from utils.config_util import ConfigDict
# from configs.im2latex import config  # todo argparse指定配置文件
# from configs.text_recognition import config  # 如果这里指定了config， 如何在import时候导入到不同的组件中


if __name__ == '__main__':

    args = ConfigDict(config)  # config file从文件读入
    visualizer = Visualizer()
    train_loader = build_dataset(args)
    train_size = len(train_loader)
    print('#batch images = %d' % train_size)
    if args.require_evaluation:
        val_loader = build_dataset(args, isTrain=False)
        val_size = len(val_loader)
        evaluator = build_evaluator(args)

    model = build_model(args)  # 先调用了initialize，
    model.setup(args)  # 在base_model 里面有一些操作
    total_steps = 0

    for epoch in range(args.epoch_count, args.epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_steps += 1
            epoch_iter += 1
            model.set_input(data)  # set_input
            model.optimize_parameters()

            if total_steps % args.print_freq == 0:
                losses = model.get_current_losses()  # Base model的方法
                t = time.time() - iter_start_time
                model.print_train_info(epoch, epoch_iter, t, t_data)

                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_steps % args.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            if args.require_evaluation and total_steps % args.eval_iter_freq == 0:
                print('evaluation the model at the end of epoch %d, iters %d' %
                      (epoch, epoch_iter))
                evaluator(model, val_loader)

            iter_data_time = time.time()
        if epoch % args.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        if args.require_evaluation and epoch % args.eval_epoch_freq == 0:
            print('evaluation the model at the end of epoch %d' % (epoch))
            evaluator(model, val_loader)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, args.epoch, time.time() - epoch_start_time))
        model.update_learning_rate()
