"""
还是遵循一切中间数据不落地的原则，避免造成额外的磁盘压力，也使得每个epoch的数据不相同
"""
import os
import cv2
import torch.utils.data as data
import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from utils.tps_grid_gen import TPSGridGen
from utils.preprocess_util import correct_image


label_prefix = 'labels'


def right_warp(fiducial_point_src, start_position, end_position, max_diverge, is_up=True):
    step = int(max_diverge / (end_position - start_position + 1))
    if is_up:
        step = -step
    fiducial_point_dst = fiducial_point_src.copy()
    for i, x_index in enumerate(range(start_position, end_position+1)):  # endpoint 不可以取到
        fiducial_point_dst[:, x_index, 1] = fiducial_point_dst[:, x_index, 1] + step * (i + 1)
    return fiducial_point_dst


def left_warp(fiducial_point_src, start_position, end_position, max_diverge, is_up=True):
    step = int(max_diverge / (end_position - start_position + 1))
    if is_up:
        step = -step
    fiducial_point_dst = fiducial_point_src.copy()
    for i, x_index in enumerate(range(end_position, start_position-1, -1)):  # start_position 可以取到
        fiducial_point_dst[:, x_index, 1] = fiducial_point_dst[:, x_index, 1] + step * (i + 1)
    return fiducial_point_dst


def centre_warp(fiducial_point_src, start_position, centre_position, end_position, max_diverge, is_up=True):
    step_left = int(max_diverge / (centre_position - start_position + 1))
    step_right = int(max_diverge / (end_position - centre_position + 1))
    if is_up:
        step_left = -step_left
        step_right = -step_right
    fiducial_point_dst = fiducial_point_src.copy()

    # 先左侧
    for i, x_index in enumerate(range(start_position, centre_position+1)):  # start_position 可以取到
        fiducial_point_dst[:, x_index, 1] = fiducial_point_dst[:, x_index, 1] + step_left * (i + 1)

    # 再右侧
    for i, x_index in enumerate(range(end_position, centre_position, -1)):  # start_position 可以取到
        fiducial_point_dst[:, x_index, 1] = fiducial_point_dst[:, x_index, 1] + step_right * (i + 1)

    return fiducial_point_dst


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
        x_min, y_min, x_max, y_max = map(lambda x: int(float(x)), line)  # 原坐标点是顺时针绕一圈
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def rotate_bound(image, rotate_angle, pad_value=(0, 0, 0)):
    """
    旋转图片，超出原图大小的边界用指定颜色填充.
    :param image: 待旋转图片
    :param rotate_angle:  旋转角度
    :param pad_value: 边界填充颜色
    :return:
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rotate_angle, 1.0)

    return cv2.warpAffine(image, M, (w, h), borderValue=pad_value)


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


def norm_fiducial_point(point, w=1024, h=512):
    point = point.reshape(-1, 2)
    cen_w = w/2
    cen_h = h/2
    point[:, 0] = (point[:, 0] - cen_w)/cen_w
    point[:, 1] = (point[:, 1] - cen_h)/cen_h
    return point


def convert_np2tensor(img_numpy):
    img_numpy = img_numpy.astype(np.float32).transpose(2, 0, 1)
    return torch.tensor(img_numpy).unsqueeze(dim=0)


class TPSDataset(data.Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super(TPSDataset, self).__init__()
        self.opt = opt
        img_files = []
        exts = ['png', 'jpg', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(os.path.join(opt.dataroot, "images")):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        img_files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(img_files)))
        self.img_files = img_files
        fiducial_point_x = np.arange(32, 1024 - 32 + 1, 64).astype(np.float32)  # 一共有16个点
        fiducial_point_y = np.arange(32, 512 - 32 + 1, 64).astype(np.float32)  # 共有8个，   16*8 = 128个点基点
        xx, yy = np.meshgrid(fiducial_point_x, fiducial_point_y)
        self.fiducial_point_src = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]), axis=-1)
        self.tps = TPSGridGen(512, 1024, torch.Tensor(norm_fiducial_point(self.fiducial_point_src.copy())))  # 只是生成这个比较耗时，已经被原作者写到外面去了

    def __len__(self):
        return len(self.img_files) * 1000  # todo 为了测试少量数据临时更改

    def __getitem__(self, index):
        while True:
            im_fn = self.img_files[index // 1000]  # todo 为了测试少量数据临时更改
            img_whole = cv2.imread(im_fn)
            # img_whole, angle = correct_image(img_whole)
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
            if random.random() > 0.3:
                bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxs]
                bboxs_h.sort()  # inplace sort
                median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数

                # 缩放后的文字高度应该在15~70之间
                ratio = random.uniform(13 / median, 50 / median)
                img_whole = cv2.resize(img_whole, None, fx=ratio, fy=ratio)
                bboxs = [[bbox[0] * ratio, bbox[1] * ratio, bbox[2] * ratio, bbox[3] * ratio] for bbox in bboxs]  # 对应坐标进行转换

            # 最后我要裁剪到多大呢? 实际题块的平均尺寸， 约为 w=1500. 比较担心LSTM层的加入对长度的影响
            # 这个尺寸仍然是一个较大的尺寸， 1536 * 512  为512*512的三倍
            croped_size = (1024, 512)  # w*h
            img_croped, bboxes_target = random_crop(img_whole, bboxs, croped_size)
            if len(bboxes_target) <= 0:
                index = random.randint(0, len(self.img_files) - 1)
                continue

            """
            进行TPS形变
            """
            ran = random.random()
            max_diverge = random.randint(5, 40)
            is_up = random.choice([True, False])
            if ran < 0.3:
                # 左侧
                end_position = random.randint(0, 5)
                fiducial_point_dst = left_warp(self.fiducial_point_src, 0, end_position, max_diverge, is_up)

            elif 0.3 <= ran < 0.6:
                # 右侧旋转
                start_position = random.randint(10, 15)
                fiducial_point_dst = left_warp(self.fiducial_point_src, start_position, 15, max_diverge, is_up)
            elif 0.6 <= ran < 0.9:
                # 中间
                start_position = random.randint(3, 12)
                end_position = random.randint(start_position, 12)
                centre_position = random.randint(start_position, end_position)
                fiducial_point_dst = centre_warp(self.fiducial_point_src, start_position, centre_position, end_position, max_diverge, is_up=True)
            else:
                fiducial_point_dst = self.fiducial_point_src.copy()

            source_coordinate = self.tps(torch.Tensor(norm_fiducial_point(fiducial_point_dst.copy())).unsqueeze(dim=0))  # 目标是可以unsqueeze
            trans_grid = source_coordinate.view(1, 512, 1024, 2)

            warped_t = F.grid_sample(convert_np2tensor(img_croped), trans_grid, padding_mode='border')
            return warped_t, torch.tensor(fiducial_point_dst-self.fiducial_point_src)

    @staticmethod
    def name():
        return 'TPSDataset'

    @staticmethod
    def collate_fn(batch):
        imgs = []
        points = []
        for img, point in batch:
            imgs.append(img)
            points.append(point.permute(2, 0, 1).unsqueeze(0))
        return torch.cat(imgs, dim=0), torch.cat(points, dim=0)


if __name__ == '__main__':
    import torch

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'

    def save_tensor_img(img_t, file_name):
        img = img_t.detach().cpu().numpy().squeeze().astype(np.int)
        img = img.transpose((1, 2, 0))
        cv2.imwrite(file_name + '.png', img)

    class Opt():
        dataroot = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/text_images/'
    opt = Opt()

    data_loader = torch.utils.data.DataLoader(
        TPSDataset(opt), shuffle=False, batch_size=1, num_workers=1,
        collate_fn=TPSDataset.collate_fn)
    # 注意这样得到的通道数在最后，默认的
    for ii, (warped_img, fiducial_point) in enumerate(data_loader):
        save_tensor_img(warped_img, os.path.join(save_path, str(ii)))
        print(ii)