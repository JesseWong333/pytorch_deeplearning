# encoding: utf-8
# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/9/20

import sys
sys.path.append("..")
import cv2
from tools import InferModel
import random
from utils.config_util import ConfigDict

# images = []
# formulas = []
# image = transforms.ToTensor()(Image.fromarray(img))
# print(image.shape)
# img = np.ones((image.shape[0], image.shape[1], image.shape[2]))
# img[:, 0:image.shape[1], 0:image.shape[2]] = image
# images.append(img[np.newaxis, :, :, :])
# images = np.concatenate(images, axis=0)
# images = images.astype(np.float32)
# formulas.append('')
# input_data = (images, formulas)

if __name__ == '__main__':
    # todo: 原先的im2latex部分参数通过yaml配置, 并没有整合
    from configs.im2latex import config
    # args = ConfigDict(config)
    # args.isTrain = False
    # print(load_config(args.im2latex_congigs))
    im2latex_args = ConfigDict(config)
    im2latex_args.isTrain = False
    im2latex_model = InferModel(im2latex_args)
    # args.update(im2latex_args)

    testfile = '/home/chen/hcn/data/formula/formula_images/labels/0917/all_test.txt'
    allines = open(testfile, 'r', encoding='utf-8').read().strip('\n').split('\n')
    count = 0
    num = 0
    allines = allines[1:]
    random.shuffle(allines)
    for line in allines:
        num += 1
        img_path, latex = line.split(' ', 1)
        img_path = img_path.replace('/media', '/home')
        img = cv2.imread(img_path, 0)
        d = im2latex_model.infer(img)
        if d == latex:
            count += 1
        if not num == 0 and num % 100 == 0:
            print('right num:{} / all num:{} accuracy:{}'.format(count, num, count/num))


