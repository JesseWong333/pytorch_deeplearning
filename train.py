import time
from options.base_options import BaseOptions
from data import create_data_loader
from models import create_model

if __name__ == '__main__':
    opt = BaseOptions().parse()
    data_loader = create_data_loader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)  # 先调用了initialize，
    model.setup(opt)  # 在base_model 里面有一些操作
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # set_input
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()  # Base model的方法
                t = (time.time() - iter_start_time) / opt.batch_size
                model.print_train_info(epoch, epoch_iter, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
