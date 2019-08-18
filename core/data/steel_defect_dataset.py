import os
import cv2
import torch.utils.data as data
import random
import numpy as np
import torch
import math
from torchvision import transforms


class SteelDefectDataset(data.Dataset):

    def __init__(self, csv_file, floder):
        super(SteelDefectDataset, self).__init__()
        with open(csv_file, 'r', encoding='utf-8') as f:
            f.readline()
            self.lines = f.readlines()
        self.floder = floder
        self.transfrom = transforms.Compose([
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.lines)

    def create_heatmap(self):
        pass

    def __getitem__(self, index):
        index_r = index // 2
        image_o = index % 2
        if image_o == 0:
            ordinal = 's1'
        else:
            ordinal = 's2'
        line_info = self.lines[index_r].strip().split(',')

        img_path = os.path.join(self.img_base_path, line_info[1], 'Plate' + line_info[2],
                                line_info[3] + '_' + ordinal + '_')
        images_channel = []
        for i in range(1, 6 + 1):
            img_path_channel = img_path + 'w' + str(i) + '.png'
            img = cv2.imread(img_path_channel, -1)
            images_channel.append(img[:, :, np.newaxis])

        img = np.concatenate(images_channel, axis=2)
        img = self.transfrom(img)

        label = int(line_info[4])
        return img, torch.tensor(label)

    @staticmethod
    def collate_fn(batch):
        imgs = []
        targets = []
        for img, target in batch:
            imgs.append(img.unsqueeze(dim=0))
            targets.append(target.unsqueeze(dim=0))
        return torch.cat(imgs, 0), torch.cat(targets, 0)


if __name__ == '__main__':
    import torch
    import time
    np.random.seed(0)
    torch.manual_seed(0)

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'

    class Opt():
        dataroot = '/media/Data/open_images_datasets/recursion-cellular-image-classification/train.csv'
    opt = Opt()

    data_loader = torch.utils.data.DataLoader(
        KaggleDataset(opt), shuffle=True, batch_size=4, num_workers=4,
        collate_fn=KaggleDataset.collate_fn)
    # 注意这样得到的通道数在最后，默认的
    start_time = time.time()
    for ii, (image, label) in enumerate(data_loader):
        # image = np.squeeze(image, axis=0)
        # image = image.transpose((1, 2, 0))
        # new_im = image.copy()
        # new_im = new_im.astype(np.uint8)
        # print(image[0,0,0,0].numpy())
        # print(label[0].numpy())
        print(ii)
        if ii == 500:
            print(time.time() - start_time)
            break