import os
import cv2
import numpy as np
from tools import InferModel
from utils.config_util import ConfigDict


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


if __name__ == '__main__':
    from configs.c2td import config
    args = ConfigDict(config)
    args.isTrain = False

    img_base_path = '/media/Data/wangjunjie_code/advancedEast/dataset/junior_biology'

    c2td_model = InferModel(args)

    for img, file_name in img_generator(img_base_path):
        h, w, _ = img.shape
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        d = c2td_model.infer(img)
        pass

