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
from core.data.utils import load_ann_file


@register_dataset
class TextDetectionDataset(data.Dataset):

    def __init__(self, dataroot, collect_keys, pipeline, isTrain=True, **kwargs):
        super(TextDetectionDataset, self).__init__()
        self.dataroot = dataroot
        self.img_files, self.trg_list = load_ann_file(self.dataroot)
        self.isTrain = isTrain
        self.pipeline = Compose(pipeline)
        # self.dataroot = osp.join(dataroot, data_prefix)
        self.collect_keys = collect_keys

    def __len__(self):
        return len(self.img_files)

    def prepare_train_img(self, idx):
        img_file = self.img_files[idx]
        ann_file = osp.join(self.dataroot, "labels", str(osp.basename(img_file).split(".")[0]) + ".txt")
        # print("ann_file", ann_file)
        results = dict(filename=img_file, ann_info=ann_file)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_file = self.img_files[idx]
        ann_file = osp.join(self.dataroot, "labels", str(osp.basename(img_file).split(".")[0]) + ".txt")
        results = dict(filename=img_file, ann_info=ann_file)
        return self.pipeline(results)

    def __getitem__(self, idx):
        # 如果当前的index有问题，需要继续随机的找一个
        if self.isTrain:
            while True:
                data = self.prepare_train_img(idx)
                if data is None:
                    idx = random.randint(0, len(self.img_files) - 1)
                    continue
                return [data[key] for key in self.collect_keys]
        else:
            data = self.prepare_test_img(idx)
            return [data[key] for key in self.collect_keys]
            
    @staticmethod
    def name():
        return 'TextDetectionDataset'

    @staticmethod
    def collate_fn(batch):
        pass
        # imgs = []
        # heat_maps = []
        # print("batch", len(batch[0]))
        # for img, heat_map in batch:
        #     # img = img.transpose((2, 0, 1))  # 通道前置
        #     # print(img.shape)
        #     imgs.append(img[np.newaxis, :, :, :])
        #     # print(heat_map.shape)
        #     heat_maps.append(heat_map[np.newaxis, :, :])
        # return np.concatenate(imgs, axis=0), np.concatenate(heat_maps, axis=0)


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