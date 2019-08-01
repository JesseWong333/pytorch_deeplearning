import time
from core.data import build_dataset
from core import build_model
from configs.config_util import ConfigDict
from configs.c2td import config

if __name__ == '__main__':
    args = ConfigDict(config)  # config file从文件读入
    data_loader = build_dataset(args)
    dataset_size = len(data_loader)
    print('#batch images = %d' % dataset_size)

    model = build_model(args)  # 先调用了initialize，
    model.setup(args)  # 在base_model 里面有一些操作
    total_steps = 0

    for epoch in range(args.epoch_count, args.epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_steps += args.batch_size
            epoch_iter += args.batch_size
            model.set_input(data)  # set_input
            model.optimize_parameters()

            if total_steps % args.print_freq == 0:
                losses = model.get_current_losses()  # Base model的方法
                t = (time.time() - iter_start_time) / args.batch_size
                model.print_train_info(epoch, epoch_iter, t, t_data)

            if total_steps % args.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % args.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, args.epoch, time.time() - epoch_start_time))
        model.update_learning_rate()
