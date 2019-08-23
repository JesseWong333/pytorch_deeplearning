# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/2

import os
import cv2
import numpy as np
import torchvision.transforms.functional as F
import operator
from PIL import ImageDraw, Image, ImageFont
from tools import InferModel
from configs.config_util import ConfigDict


"""
检测 + 识别 pipeline
pipeline 要抽象吗？ 使用多线程机制
用于连接起来测试
-- 这样写完全还是一团糟~~
"""


# 重新写的一个简要的可视化，并没有很好的考虑排序，重叠的问题
def visualize(img, bounding_boxes, labellist, img_patches, font, file_name, save_path):
    h, w, _ = img.shape

    ori_img = Image.fromarray(img)
    ori_img = ori_img.convert('RGB')

    newimg = Image.new('RGB', (int(w * 3.3), int(h * 1.2)), (225, 225, 225))
    newimg.paste(ori_img, (0, 0))

    txt_draw = ImageDraw.Draw(newimg)

    for cords, text, img_patch in zip(bounding_boxes, labellist, img_patches):
        # 传出的接口没有统一
        if isinstance(cords, list):
            xmin, ymin = cords[0]
            cords_top = cords[0:-1:2]
            cords_down = cords[-1:0:-2]  # 要逆序
            cords_sorted = cords_top + cords_down + [[xmin, ymin]]
            cords_sorted = [(cord[0], cord[1]) for cord in cords_sorted]
            txt_draw.line(cords_sorted, width=2, fill='blue')
        else:
            xmin, ymin = cords[0:2]
            xmax, ymax = cords[2:4]
            txt_draw.line([(xmin, ymin),
                           (xmax, ymin),
                           (xmax, ymax),
                           (xmin, ymax),
                           (xmin, ymin)], width=2, fill='red')

        txt_draw.text((xmin+2*w, ymin), text, fill=6, font=font)
        img_patch = Image.fromarray(img_patch).convert('RGB')
        newimg.paste(img_patch, (xmin+w, ymin))
    newimg.save(os.path.join(save_path, file_name + '.png'))


class PipeLine(object):
    def __init__(self):
        from configs.c2td import config as c2td_config
        from configs.crnn import config as crnn_config
        c2td_args = ConfigDict(c2td_config)
        c2td_args.isTrain = False
        crnn_args = ConfigDict(crnn_config)
        crnn_args.isTrain = False

        self.c2td_model = InferModel(c2td_args)
        self.crnn_model = InferModel(crnn_args)

    def infer(self, img):
        h, w, _ = img.shape
        # 检测填充
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        img_patches, bounding_boxes = self.c2td_model.infer(img)  # 如果是扭曲的，这里应该是返回校正后的图片，为了接口统一，检测统一返回校正后的图片

        nh = 32  # todo magic number
        texts = []
        for img_patch in img_patches:  # 是否可以并行
            # 这个版本的crnn要求高度为32的灰度图, 且需要进行输入图片的归一化
            gray_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            h, w = gray_patch.shape
            nw = int(nh * w / h)
            gray_patch = cv2.resize(gray_patch, (nw, nh))
            gray_patch = np.expand_dims(gray_patch, axis=2)
            gray_patch = F.to_tensor(gray_patch)  # 这个标准步骤我以后还是做一下比较好，进行数据归一化
            gray_patch.sub_(0.5).div_(0.5)
            text = self.crnn_model.infer(gray_patch)
            texts.append(text)

        assert len(texts) == len(bounding_boxes)

        return bounding_boxes, texts, img_patches


if __name__ == '__main__':
    from tqdm import tqdm

    def img_generator(folder):
        file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
        img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            file_name = img_path.split('/')[-1].split('.')[0]
            yield img, file_name

    pipeline = PipeLine()

    img_base_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo/'
    # img_base_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo/all-images/'
    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo_results'

    font_ttf = "STKAITI.TTF"  # 可视化字体类型
    font = ImageFont.truetype(font_ttf, 32)  # 字体与字体大小

    for img, file_name in tqdm(img_generator(img_base_path)):
        bounding_boxes, texts, img_patches = pipeline.infer(img)
        visualize(img, bounding_boxes, texts, img_patches, font, file_name, save_path)


