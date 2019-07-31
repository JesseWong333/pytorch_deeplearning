"""
推理测试同样采取通用的方式，但是相比参考的项目应该减少参数
不同的方法可能会有不同的后处理
"""

# todo 没写完，怎么解耦. 关键是 xxx-model里面的set_input 部分, 通过is train 部分判断， 测试阶段就只送一个

import os
from options.base_options import BaseOptions
import cv2
import numpy as np
import torch
from core import create_model
from utils.preprocess_util import correct_image


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img, angle = correct_image(img)  # 校正模块
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


if __name__ == '__main__':
    # 其实在bese_model中集成了很多方法，装载模型，切换为eval等，都非常好
    img_base_path = '/media/Data/wangjunjie_code/advancedEast/dataset/junior_biology'
    opt = BaseOptions().parse()
    gpu_id = 0
    opt.gpu_ids = [gpu_id]
    device = torch.device('cuda:{}'.format(gpu_id))
    model = create_model(opt)  # 先调用了initialize，
    model.setup(opt)
    model.load_networks(100)
    model.eval()
    model.test()
    # 哪里控制送到那个GPU呢？ 在model方法中的initialize控制了送的位置

    for img, file_name in img_generator(img_base_path):
        h, w, _ = img.shape
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        h, w, _ = img.shape

        img_t = img.transpose((2, 0, 1)).astype(np.float32)  # 通道前置

        img_t = torch.from_numpy(img_t)
        img_t = img_t.unsqueeze(dim=0).to(device)

        model.set_input(img_t)
        model.forward()
        out = model.score

        # 进行后处理
