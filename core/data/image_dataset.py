import os
import cv2
import torch.utils.data as data
import random
import numpy as np


label_prefix = 'labels'


def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line)  # 原坐标点是顺时针绕一圈
        x_min = min(x1, x4)
        y_min = min(y1, y2)
        x_max = max(x2, x3)
        y_max = max(y3, y4)
        bbox.append([x_min, y_min, x_max, y_max])
    return bbox


def load_annoataion_quard(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")[:4]
        x_min, y_min, x_max, y_max = map(int, line)  # 原坐标点是顺时针绕一圈
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def cal_bbox(x_min, y_min, x_max, y_max, bboxs):
    bboxs_target = []
    for bbox in bboxs:
        # 高度方面，只要出现1/3在外面就不检测， 宽度方面， 只要超过一个h就要检测
        tol_pix_h = (bbox[3]-bbox[1]) / 3
        tol_pix_w = bbox[3]-bbox[1]

        c_xmin = bbox[0] - x_min
        c_ymin = bbox[1] - y_min
        c_xmax = bbox[2] - x_min
        c_ymax = bbox[3] - y_min

        # 不检测的类型， 高度方面，有1/3在外面
        if not (-tol_pix_h < c_ymin < y_max-y_min+tol_pix_h and -tol_pix_h < c_ymax < y_max-y_min + tol_pix_h):
            continue

        # 宽度判断
        in_xmin = max(0, c_xmin)
        in_xmax = min(x_max-x_min, c_xmax)
        if in_xmax - in_xmin < tol_pix_w:  # 小于0完全在外面
            continue

        if c_xmin < 0:
            c_xmin = 0.
        if c_ymin < 0:
            c_ymin = 0.
        if c_xmax > x_max - x_min:
            c_xmax = x_max - x_min
        if c_ymax > y_max - y_min:
            c_ymax = y_max - y_min
        bboxs_target.append([c_xmin, c_ymin, c_xmax, c_ymax])
    return bboxs_target


def random_crop(img, bboxs, size):
    h, w, c = img.shape  # 最后再确定到底是用三通道，还是灰度图

    # 只在右下角补, bboxs坐标不变
    if w <= size[0]:
        img = np.pad(img, ((0, 0), (0, size[0]-w + 5), (0, 0)), 'constant', constant_values=255)
    if h <= size[1]:
        img = np.pad(img, ((0, size[1]-h + 5), (0, 0), (0, 0)), 'constant', constant_values=255)

    h, w, c = img.shape
    x_min = random.randint(0, w - size[0] - 1)  # randint后面可以取到
    y_min = random.randint(0, h - size[1] - 1)
    x_max = x_min + size[0]
    y_max = y_min + size[1]
    img_croped = img[y_min: y_max, x_min: x_max]
    bboxs_target = cal_bbox(x_min, y_min, x_max, y_max, bboxs)
    return img_croped, bboxs_target


class ImageDataset(data.Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(ImageDataset, self).__init__()
        self.opt = opt
        img_files = []
        exts = ['png', 'jpg', 'jpeg', 'JPG']  # todo : 这里暂时只使用合成的数据. 真实的数据都是png格式的
        for parent, dirnames, filenames in os.walk(os.path.join(opt.dataroot, "images")):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        img_files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(img_files)))
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 如果当前的index有问题，需要继续随机的找一个
        while True:
            im_fn = self.img_files[index]
            img_whole = cv2.imread(im_fn)
            # h, w, c = img_whole.shape
            # im_info = np.array([h, w, c]).reshape([1, 3])  # 3*1 变为1*3

            _, fn = os.path.split(im_fn)
            fn, _ = os.path.splitext(fn)
            txt_fn = os.path.join(self.opt.dataroot, label_prefix, fn + '.txt')
            if not os.path.exists(txt_fn):
                # print("Ground truth for image {} not exist!".format(im_fn))
                index = random.randint(0, len(self.img_files) - 1)
                continue
            bboxs = load_annoataion_quard(txt_fn)
            if len(bboxs) == 0:
                # print("Ground truth for image {} empty!".format(im_fn))
                index = random.randint(0, len(self.img_files) - 1)
                continue

            # 增广操作
            # 以一定的概率进行增广
            if random.random() > 0.5:
                bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxs]
                bboxs_h.sort()  # inplace sort
                median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数

                # 缩放后的文字高度应该在15~70之间
                ratio = random.uniform(13 / median, 70 / median)
                img_whole = cv2.resize(img_whole, None, fx=ratio, fy=ratio)
                bboxs = [[bbox[0] * ratio, bbox[1] * ratio, bbox[2] * ratio, bbox[3] * ratio] for bbox in bboxs]  # 对应坐标进行转换

            # 最后我要裁剪到多大呢? 实际题块的平均尺寸， 约为 w=1500. 比较担心LSTM层的加入对长度的影响
            # 这个尺寸仍然是一个较大的尺寸， 1536 * 512  为512*512的三倍
            croped_size = (1024, 512)  # w*h
            img_croped, bboxes_target = random_crop(img_whole, bboxs, croped_size)
            if len(bboxes_target) <= 0:
                index = random.randint(0, len(self.img_files) - 1)
                continue

            # 创建对应的heat map作为label, 1/4 map_size
            factor = 4
            heat_map = np.zeros((croped_size[1] // factor, croped_size[0] // factor), dtype=float)
            for bbox in bboxes_target:
                bbox = [int(cond // factor) for cond in bbox]
                heat_map[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1.

            # 将bboxes_target 转换为 头 尾 的形式
            bboxes_silde = []
            for bbox in bboxes_target:
                x_min, y_min, x_max, y_max = bbox
                bboxes_silde.append([x_min, y_min, x_min + (y_max-y_min), y_max, 1])
                bboxes_silde.append([x_max - (y_max-y_min), y_min, x_max, y_max, 2])

            return img_croped, bboxes_silde, heat_map

    @staticmethod
    def name():
        return 'ImageDataset'

    @staticmethod
    def collate_fn(batch):
        imgs = []
        bboxs_batch = []
        heat_maps = []
        for img, bboxs, heat_map in batch:
            img = img.transpose((2, 0, 1))  # 通道前置
            imgs.append(img[np.newaxis, :, :, :])
            heat_maps.append(heat_map[np.newaxis, :, :])
            bboxs_batch.append(np.array(bboxs).astype(np.float32))
        return np.concatenate(imgs, axis=0), bboxs_batch, np.concatenate(heat_maps, axis=0)


if __name__ == '__main__':
    import torch

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'

    class Opt():
        dataroot = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
    opt = Opt()

    data_loader = torch.utils.data.DataLoader(
        ImageDataset(opt), shuffle=False, batch_size=1, num_workers=1,
        collate_fn=ImageDataset.collate_fn)
    # 注意这样得到的通道数在最后，默认的
    for ii, (image, bboxs, heatmap) in enumerate(data_loader):
        image = np.squeeze(image, axis=0)
        image = image.transpose((1, 2, 0))
        new_im = image.copy()
        # new_im = new_im.astype(np.uint8)
        bboxs = bboxs[0].astype(np.int)

        for bbox in bboxs:
            cv2.rectangle(new_im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        b = cv2.imwrite(os.path.join(save_path, str(ii) + '.png'), new_im)
        print(ii)