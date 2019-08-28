import os
import cv2
import torch.utils.data as data
import random
import numpy as np
import torch
import math
from torchvision import transforms
import pandas as pd
from . import register_dataset


@register_dataset
class SteelDefectDataset(data.Dataset):

    def __init__(self, csv_file, folder):
        super(SteelDefectDataset, self).__init__()
        self.train_df = pd.read_csv(csv_file)

        self.folder = folder
        self.transfrom = transforms.Compose([
            transforms.ToTensor(),
            ])

    # https://www.kaggle.com/go1dfish/clear-mask-visualization-and-simple-eda
    def name_and_mask(self, start_idx):
        col = start_idx
        img_names = [str(i).split("_")[0] for i in self.train_df.iloc[col:col+4, 0].values]
        if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
            raise ValueError

        labels = self.train_df.iloc[col:col+4, 1]
        mask = np.zeros((256, 1600, 4), dtype=np.uint8)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                mask_label = np.zeros(1600*256, dtype=np.uint8)
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                for pos, le in zip(positions, length):
                    mask_label[pos:(pos+le)] = 1
                mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

        # 已知是没有重叠的
        # 第0维，是空白的维度.
        mask_blank = np.ones((256, 1600), dtype=np.uint8)
        for i in range(4):
            mask_blank = mask_blank - mask[:, :, i]
        mask_blank[mask_blank < 0] = 0
        mask = np.concatenate((mask_blank[:, :, np.newaxis], mask), 2)

        # 合并，转换为一个map， 使用标签[0, 4]
        for i in range(5):
            mask[mask[:, :, i] == 1, i] = i

        mask_class = np.zeros((256, 1600), dtype=np.uint8)
        for i in range(5):
            mask_class += mask[:, :, i]
        return img_names[0], mask_class

    def __len__(self):
        return self.train_df.shape[0]//4

    def __getitem__(self, index):
        col = index * 4
        img_name, mask = self.name_and_mask(col)
        img = cv2.imread(os.path.join(self.folder, img_name), -1)
        return img, mask

    @staticmethod
    def collate_fn(batch):
        imgs = []
        targets = []
        for img, target in batch:
            img = img.transpose((2, 0, 1))  # 通道前置
            imgs.append(img[np.newaxis, :, :, :])
            # target = target.transpose((2, 0, 1))
            targets.append(target[np.newaxis, :, :])
        return np.concatenate(imgs, 0), np.concatenate(targets, 0)


if __name__ == '__main__':
    import torch
    import time
    np.random.seed(0)
    torch.manual_seed(0)

    csv_file = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/severstal-steel-defect-detection/train.csv'
    folder = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/severstal-steel-defect-detection/train_images'
    data_loader = torch.utils.data.DataLoader(
        SteelDefectDataset(csv_file, folder), shuffle=True, batch_size=4, num_workers=4,
        collate_fn=SteelDefectDataset.collate_fn)
    # 注意这样得到的通道数在最后，默认的
    start_time = time.time()
    for ii, (image, label) in enumerate(data_loader):
        print(ii)
        if ii == 500:
            print(time.time() - start_time)
            break