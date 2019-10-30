import random
import cv2
import math
import numpy as np
# from utils.preprocess_util import correct_image

def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, line[:8])  # 原坐标点是顺时针绕一圈
        if (y3 - y1) < 10:
            continue
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


def create_heat_map_patch(xmin, ymin, xmax, ymax):
    """
    :param xmin: 除以4之后的小数点坐标
    :param ymin:
    :param xmax:
    :param ymax:
    :return:
    """
    xmax = xmax - xmin
    ymax = ymax - ymin

    # 先生成 ymax * 1 * 1 大小的片段， 再扩充成 ymax * xmax * 3 大小的feature map
    line_pach = np.zeros((ymax, 1), dtype=float)

    if ymax % 2 == 0:
        # ymax为偶，长度是为奇数
        centre = int(ymax / 2)

        step_len = 1./centre/2
        for step, p in enumerate(range(centre, -1, -1)):  # 取不到 -1
            line_pach[p, 0] = 1. - step*step_len

        for step, p in enumerate(range(centre, ymax, 1)):
            line_pach[p, 0] = 1. - step*step_len
    else:
        # ymax为奇数，长度是为偶数
        centre = math.floor(ymax / 2)
        step_len = 1. / centre / 2
        for step, p in enumerate(range(centre, -1, -1)):  # 取不到 -1
            line_pach[p, 0] = 1. - step*step_len

        for step, p in enumerate(range(centre+1, ymax, 1)):
            line_pach[p, 0] = 1. - step*step_len

    return np.tile(line_pach, (1, xmax))


def cal_bbox(x_min, y_min, x_max, y_max, bboxs):
    bboxs_target = []
    for bbox in bboxs:
        # 高度方面，只要出现1/3在外面就不检测， 宽度方面， 只要超过一个h就要检测
        tol_pix_h = (bbox[3]-bbox[1]) / 10 * 8
        tol_pix_w = bbox[2]-bbox[0] / 10 * 2

        c_xmin = bbox[0] - x_min
        c_ymin = bbox[1] - y_min
        c_xmax = bbox[2] - x_min
        c_ymax = bbox[3] - y_min

        # 不检测的类型， 高度方面，有1/3在外面
        # if not (-tol_pix_h < c_ymin < y_max-y_min+tol_pix_h and -tol_pix_h < c_ymax < y_max-y_min + tol_pix_h):
        #     continue

        # 宽度判断
        in_xmin = max(0, c_xmin)
        in_xmax = min(x_max-x_min, c_xmax)
        if in_xmax - in_xmin <= 10:  # 小于0完全在外面
            continue

        #高度判断
        in_ymin = max(0, c_ymin)
        in_ymax = min(y_max - y_min, c_ymax)
        if in_ymax - in_ymin <= 10:
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

def random_scale(img_whole, bboxs):
    bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxs]
    bboxs_h.sort()  # inplace sort
    median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数

    # 缩放后的文字高度应该在15~70之间
    if median == 0:
        median = 35
    ratio = random.uniform(15 / median, 140 / median)

    img_whole = cv2.resize(img_whole, None, fx=ratio, fy=ratio)
    bboxes = [[bbox[0] * ratio, bbox[1] * ratio, bbox[2] * ratio, bbox[3] * ratio] for bbox in bboxs]  # 对应坐标进行转换
    return img_whole, bboxes


def random_scale_cn(img_whole, bboxs):
    # bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxs]
    # bboxs_h.sort()  # inplace sort
    # median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数
    #
    # # 缩放后的文字高度应该在15~70之间
    # if median == 0:
    #     median = 35
    # ratio = random.uniform(15 / median, 140 / median)
    ratio = random.uniform(0.5, 2)
    img_whole = cv2.resize(img_whole, None, fx=ratio, fy=ratio)
    bboxes = [[bbox[0] * ratio, bbox[1] * ratio, bbox[2] * ratio, bbox[3] * ratio] for bbox in bboxs]  # 对应坐标进行转换
    return img_whole, bboxes


def random_scale_cn2(img_whole, bboxs):
    # bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxs]
    # bboxs_h.sort()  # inplace sort
    # median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数
    #
    # # 缩放后的文字高度应该在15~70之间
    # if median == 0:
    #     median = 35
    w_ratio = random.uniform(0.3, 2)
    h_ratio = random.uniform(0.5, 2)
    img_whole = cv2.resize(img_whole, None, fx=w_ratio, fy=h_ratio)
    bboxes = [[bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio] for bbox in bboxs]  # 对应坐标进行转换
    return img_whole, bboxes


def image_normalize(img, pixel_mean, std):
    img = img.astype(np.float32)
    img /= 255
    h, w, c = img.shape
    for i in range(c):
        img[:, :, i] -= pixel_mean[i]
        img[:, :, i] /= std[i]
    return img


# def img_generator(folder):
#     file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
#     img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         img, angle = correct_image(img)  # 校正模块
#         file_name = img_path.split('/')[-1].split('.')[0]
#         yield img, file_name
