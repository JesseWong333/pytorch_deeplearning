"""
add by hzc
"""

import os
import os.path as osp
import codecs
import cv2
import torch.utils.data as data
import random
import numpy as np
import math
from . import register_dataset
from .pipelines.compose import Compose
from .pipelines.transforms import Pad, Normalize
import itertools
from core.data.utils import load_ann_file



@register_dataset
class TextRecognitionDataset(data.Dataset):

    def __init__(self, ann_file, pipeline, pad_minw, collect_keys, isTrain=True, **kwargs):
        super(TextRecognitionDataset, self).__init__()
        self.dataroot = ann_file
        self.collect_keys = collect_keys
        self.img_files, self.trg_list = load_ann_file(ann_file)
        self.isTrain = isTrain
        self.pipeline = Compose(pipeline)
        # self.pipeline = Compose(pipeline)
        # self.collate_pipeline = Compose(pipeline['collate_pipeline'])
        self.pad = Pad(isTrain=isTrain, minw=pad_minw)
        self.normalize = Normalize()

    def __len__(self):
        return len(self.img_files)

    def prepare_train_img(self, idx):
        img_file = self.img_files[idx]
        cpu_text = self.trg_list[idx]
        results = dict(filename=img_file, cpu_text=cpu_text)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_file = self.img_files[idx]
        cpu_text = self.trg_list[idx]
        results = dict(filename=img_file, cpu_text=cpu_text)
        return self.pipeline(results)

    def __getitem__(self, idx):
        # 如果当前的index有问题，需要继续随机的找一个
        if self.isTrain:
            data = self.prepare_train_img(idx)
        else:
            data = self.prepare_test_img((idx))
        return data

    @staticmethod
    def name():
        return 'TextRecognitionDataset'

    def collate_fn(self, batch):
        img_shape = [i['img_shape'] for i in batch]
        maxw = max(img_shape, key=lambda x: x[1])[1]
        imgs, text, length, filename, cpu_text = [], [], [], [], []
        for _ in batch:
            _ = self.pad(_, maxw)
            _ = self.normalize(_)
            imgs.append(_['img'][np.newaxis, :, :, :])
            text.append(_['text'])
            length.append(_['length'])
            filename.append(_['filename'])
            cpu_text.append(_['cpu_text'])
        imgs = np.concatenate(imgs, axis=0)
        text = np.array(list(itertools.chain(*text)), dtype=np.int32)
        length = np.array(length, dtype=np.int32).reshape((-1, ))
        # print(self.collect_keys)
        results = dict(img=imgs, text=text, length=length, filename=filename, cpu_text=cpu_text)
        # print(imgs.shape, text.shape, length.shape, length[0])

        return [results[key] for key in self.collect_keys]





# if __name__ == '__main__':
#     import torch
#
#     save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'
#
#     class Opt():
#         dataroot = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
#     opt = Opt()
#
#     data_loader = torch.utils.data.DataLoader(
#         ImageLine2Dataset(opt), shuffle=False, batch_size=1, num_workers=1,
#         collate_fn=ImageLine2Dataset.collate_fn)
#     # 注意这样得到的通道数在最后，默认的
#     for ii, (image, heatmap) in enumerate(data_loader):
#         image = np.squeeze(image, axis=0)
#         image = image.transpose((1, 2, 0))
#         new_im = image.copy()
#         # new_im = new_im.astype(np.uint8)
#         print(ii)