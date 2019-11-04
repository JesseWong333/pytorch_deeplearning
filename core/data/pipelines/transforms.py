# -*- coding: utf-8 -*-
"""
add by hzc
添加一些常用的预处理的模块以供配置进行调用
"""

import numpy as np
import cv2
import os
import torch
import codecs
import os.path as osp
import torchvision.transforms.functional as F
# from . import register_pre_process
from torchvision import transforms
from PIL import Image
import random
import collections
import math
from . import register_pipeline


@register_pipeline
class LoadImageFromFile(object):

    def __init__(self,  gray=False):
        self.gray = gray

    def __call__(self, results):
        filename = results['filename']
        img = cv2.imread(filename, 0) if self.gray else cv2.imread(filename)
        results['img'] = img
        results['ori_shape'] = img.shape
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(gray={})'.format(
            self.gray)


@register_pipeline
class LoadAnnotations(object):
    """标注暂时只考虑两套, 一对多, 一对一"""

    def __init__(self,
                 with_bbox=True,
                #  with_textlabel=False,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        # self.with_textlabel = with_textlabel
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        bboxes = []
        with codecs.open(ann_info, "r", "utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")[:4]
            x_min, y_min, x_max, y_max = map(int, line)  # 原坐标点是顺时针绕一圈
            bboxes.append([x_min, y_min, x_max, y_max, 1])
        if len(bboxes) == 0 and self.skip_img_without_anno:
             return None
        results['bboxes'] = bboxes
        return results

    # def _load_textlabels(self, results):
    #     return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            # if results is None:
            #     return None
        # if self.with_textlabel:
        #     results = self._load_textlabels(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={})').format(self.with_bbox,
                                             )
        return repr_str


@register_pipeline
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean=None, std=None):
        self.mean = 0.5 if mean is None else np.array(mean, dtype=np.float32)
        self.std = 0.5 if std is None else np.array(std, dtype=np.float32)

    def __call__(self, results):
        # if isinstance(results, np.array):
        #     img = results
        # else:
        img = results['img']
        # if self.gray:
        #     img = np.expand_dims(img, axis=2)
        # img = F.to_tensor(img)  # 这个标准步骤我以后还是做一下比较好，进行数据归一化
        if img.ndim == 2:
            img = img[:, :, None]
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32)
        img = (img / 255 - self.mean) / self.std
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(gray={})'.format(self.gray)
        return repr_str


@register_pipeline
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self, \
                img_scale=None,
                with_textscale=None,
                with_normalscale=None, 
                with_bbox=False,
                max_pixel_limit=None,
                max_longside_limit=None,
                recog_scale=None):
        if img_scale is None:
            img_scale = dict()
            img_scale['scale_prob'] = 0.5
        self.img_scale = img_scale
        self.with_textscale = with_textscale
        self.with_normalscale = with_normalscale
        self.with_bbox = with_bbox
        self.max_pixel_limit = max_pixel_limit
        self.max_longside_limit = max_longside_limit
        self.with_ratio = True if self.with_textscale or self.with_normalscale else False
        self.recog_scale = recog_scale
 
    def _random_scale(self, results):
        # img = results['img']
        ratio = 1.0
        scale_prob = self.img_scale['scale_prob']
        if self.with_textscale:
            bboxes = results['bboxes']
            text_min_h =self.with_textscale['text_min_h']
            text_max_h =self.with_textscale['text_max_h']
            if random.random() > scale_prob:
                bboxes_h = [(bbox[3] - bbox[1]) for bbox in bboxes]
                bboxes_h.sort()  # inplace sort
                median = bboxes_h[int(len(bboxes_h) / 2)]  # 一张图片文字高度的中位数
                # 缩放后的文字高度应该在15~70之间
                ratio = random.uniform(text_min_h / median, text_max_h / median)
                self.with_ratio = True
        elif self.with_normalscale:
            if random.random() > scale_prob:
                min_ratio = self.with_normalscale['min_ratio']
                max_ratio = self.with_normalscale['max_ratio']
                ratio = random.uniform(min_ratio, max_ratio)
        results['ratio'] = ratio

    def _resize_img(self, results):
        img = results['img']
        if self.with_ratio:
            ratio = results['ratio']
            if not math.isclose(ratio, 1.0):
                img = cv2.resize(img, None, fx=ratio, fy=ratio)
        # 公式识别的resize方法
        elif self.max_pixel_limit:
            maxpixels = self.max_pixel_limit['maxpixels']
            h, w = img.shape[:2]
            if h * w >= maxpixels:  # the shape of image is too large
                nw = int(w / 4 * 3)
                nh = int(h / 4 * 3)
                while nh * nw >= maxpixels:
                    nw = int(w / 4 * 3)
                    nh = int(h / 4 * 3)
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        elif self.max_longside_limit:
            max_long_side = self.max_longside_limit
            h, w = img.shape[:2]
            if max(h, w) > max_long_side:
                if h > w:
                    nh = max_long_side 
                    nw = int(round(w * max_long_side / h)) 
                else:
                    nw = max_long_side 
                    nh = int(round(h * max_long_side / w)) 
                img = cv2.resize(img, (nh, nw), interpolatio=cv2.INTER_AREA)
        # 文字识别的resize方法
        elif self.recog_scale:
            nh = self.recog_scale['imgH']
            factor = self.recog_scale['factor']
            h, w = img.shape[:2]
            nw = int(round(nh * w / h))
            nw = nw if nw % factor == 0 else int(round(nw / factor) * factor)
            img = cv2.resize(img, (nw, nh))
            results['img_shape'] = img.shape
        results['img'] = img
        results['img_shape'] = img.shape
    
    @staticmethod
    def _resize_bboxes(results):
        img_shape = results['img'].shape
        bboxes = results['bboxes']
        ratio = results['ratio']
        if not math.isclose(ratio, 1.0):
            bboxes = [[bbox[0] * ratio, bbox[1] * ratio, bbox[2] * ratio, bbox[3] * ratio] for bbox in bboxes]  # 对应坐标进行转换
        bboxes = np.array(bboxes, dtype=np.int32)
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
        results['bboxes'] = bboxes

    def __call__(self, results):
        self._random_scale(results)
        self._resize_img(results)
        if self.with_bbox:
            self._resize_bboxes(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={})').format(self.img_scale)
        return repr_str


@register_pipeline
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size_divisor=None, pad_val=255, isTrain=True, minw=None):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.isTrain = isTrain
        self.minw = minw
        # only one of size and size_divisor should be valid

    def _pad_img(self, results, maxw=None):
        img = results['img']
        h, w = img.shape[:2]
        if isinstance(maxw, int):
            # img = results
            # h, w = img.shape[:2]
            if w < maxw:
                padding_left_w = random.randint(0, int((maxw - w))) if self.isTrain else int((maxw - w) / 2)
                img = cv2.copyMakeBorder(img, 0, 0, padding_left_w, maxw - w - padding_left_w, cv2.BORDER_CONSTANT, \
                                         value=self.pad_val)
        elif self.size_divisor is not None:

            img = np.pad(img, ((0, self.size_divisor - h % self.size_divisor), \
                               (0, self.size_divisor - w % self.size_divisor), (0, 0)), 'constant', \
                         constant_values=self.pad_val)  # 填充到16的倍数
        results['img'] = img

    def __call__(self, results, maxw=None):
        if isinstance(self.minw, int):
            maxw = maxw if maxw > self.minw else self.minw
        # return self._pad_img(results, maxw=maxw)
        self._pad_img(results, maxw=maxw)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size_divisor={}, pad_val={})'.format(
            self.size_divisor, self.pad_val)
        return repr_str


@register_pipeline
class AddNoise(object):
    def __init__(self, with_salt=None, with_blur=None):
        self.with_salt = with_salt
        self.with_blur = with_blur
     
    def _add_saltnoise(self, results):
        prob = self.with_salt['prob']
        if random.random() > prob:
            img = results['img']
            img = np.require(img, dtype='f4', requirements=['O', 'W'])
            height, width = img.shape[:2]
            rate = 1.0 * np.random.randint(0, 10) / 100.0
            for jj in range(int(width * height * rate)):
                row = np.random.randint(0, width - 1)
                col = np.random.randint(0, height - 1)
                value = np.random.randint(0, 255)
                img[col][row] = value
            img = img.astype(np.uint8)
            results['img'] = img

    def _add_blurnoise(self, results):
        prob = self.with_blur['prob']
        if random.random() > prob:
            min_val = self.with_blur['min_val']
            max_val = self.with_blur['max_val']
            img = results['img']
            rand = np.random.randint(min_val, max_val)
            img = cv2.blur(img, (rand, rand))
            img = img.astype(np.uint8)
            results['img'] = img
    
    def __call__(self, results):
        if self.with_salt is not None:
            self._add_saltnoise(results)
        if self.with_blur is not None:
            self._add_blurnoise(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(with_salt={}, with_blur={})'.format(
            self.with_salt, self.with_blur)


@register_pipeline
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, bbox_crop=None, text_crop=None):
        self.bbox_crop = bbox_crop
        self.text_crop = text_crop
    
    def _random_crop_text(self, results):
        prob = self.text_crop['prob']
        if random.random() > prob:
            img = results['img']
            # print(img.shape)
            height, width = img.shape[:2]
            padding = self.text_crop['padding']
            crop_img = np.ones(shape=[height + 2 * padding, width + 2 * padding], dtype=np.uint8) * 255
            crop_img[padding:padding + height, padding:padding + width] = img.copy()
            x1 = np.random.randint(0, padding - 1)
            y1 = np.random.randint(0, padding - 1)
            x2 = np.random.randint(0, padding - 1)
            y2 = np.random.randint(0, padding - 1)
            img = crop_img[y1:y2 + height + padding, x1:x2 + width + padding]
            results['img'] = img

    @staticmethod
    def cal_bbox(x_min, y_min, x_max, y_max, bboxes):
        bboxes_target = []
        for bbox in bboxes:
            # 高度方面，只要出现1/3在外面就不检测， 宽度方面， 只要超过一个h就要检测
            tol_pix_h = (bbox[3]-bbox[1]) / 5
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
            bboxes_target.append([c_xmin, c_ymin, c_xmax, c_ymax])
        return bboxes_target

    def _random_crop_bbox(self, results):
        img = results['img']
        size = self.bbox_crop['crop_size']
        bboxes = results['bboxes']
        h, w = img.shape[:2]  # 最后再确定到底是用三通道，还是灰度图
        # 只在右下角补, bboxs坐标不变
        if w <= size[0]:
            img = np.pad(img, ((0, 0), (0, size[0]-w + 5), (0, 0)), 'constant', constant_values=255)
        if h <= size[1]:
            img = np.pad(img, ((0, size[1]-h + 5), (0, 0), (0, 0)), 'constant', constant_values=255)
        cnt = 0
        h, w = img.shape[:2]
        bboxes_target = []
        while len(bboxes_target) < 1 and cnt < 50:
            x_min = random.randint(0, w - size[0] - 1)  # randint后面可以取到
            y_min = random.randint(0, h - size[1] - 1)
            x_max = x_min + size[0]
            y_max = y_min + size[1]
            crop_img = img[y_min: y_max, x_min: x_max]
            bboxes_target = self.cal_bbox(x_min, y_min, x_max, y_max, bboxes)
            cnt += 1
        results['img'] = crop_img
        results['bboxes'] = bboxes_target        

    def __call__(self, results): 
        if self.bbox_crop:
            self._random_crop_bbox(results)
            if len(results['bboxes']) < 1:
                return None
        elif self.text_crop:
            self._random_crop_text(results)
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_crop={}, text_crop={})'.format(
            self.bbox_crop, self.text_crop)


@register_pipeline
class GetHeatMap(object):
    """
    C2TD 生成热点图
    """

    def __init__(self, with_heatmap=None):
        self.with_heatmap = with_heatmap

    @staticmethod
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
    
    def __call__(self, results):
        factor = self.with_heatmap['factor']
        img = results['img']
        h, w = img.shape[:2]
        heat_map = np.zeros((int(h // factor), int(w // factor), 3), dtype=np.float32)
        bboxes = results['bboxes']
        for bbox in bboxes:
            bbox_4 = [int(cond / factor) for cond in bbox]
            xmin, ymin, xmax, ymax = bbox_4  # 缩放到1/4之后的坐标
            # 创建一个这么大小的feature_map, 然后拷贝过去
            try:
                heat_map_corp = self.create_heat_map_patch(xmin, ymin, xmax, ymax)
            except:
                continue
            heat_map[ymin: ymax, xmin: xmax, 0] = heat_map_corp
               # 对其中的每一个点进行标注
            for p_y in range(ymin, ymax):
                for p_x in range(xmin, xmax):
                    heat_map[p_y, p_x, 1] = bbox[1] - (p_y+0.5) * factor
                    heat_map[p_y, p_x, 2] = bbox[3] - (p_y+0.5) * factor
        results['heat_map'] = heat_map
        return results


@register_pipeline
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, voc_path, require_enc=True, ignore_case=True):
        self._ignore_case = ignore_case
        self.itos = {}
        self.stoi = {}
        with codecs.open(voc_path, 'r', 'utf-8') as f:
            data = f.read().strip()
            data = data.split()
            # print(data)
            self.itos[0] = '[blank]'
            self.stoi['[blank]'] = 0
            for i, c in enumerate(data):
                # assert c != ' '
                self.itos[i + 1] = c
                self.stoi[c] = i + 1

        print(self.itos, self.stoi)
        self.voc_len = len(self.stoi)
        self.require_enc = require_enc

    def encode(self, results):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        # hcn
        text = results['cpu_text']
        length = []
        if isinstance(text, str):
            # print("invalid input", text)
            textsplit = text.strip().split(' ')
            text = [self.stoi[c] if c in self.stoi else 0 for c in textsplit]
            length = [len(text)]
            # text = [int(item) for item in textsplit]
        elif isinstance(text, collections.Iterable):
            length = []
            newtext = np.array([])
            # max_l = max([len(t) for t in text])
            for item in text:
                # item_array = np.zeros(max_l, dtype=np.int)
                item = item.strip().split()
                item_list = [self.stoi[c] if c in self.stoi else 0 for c in item]
                # item_array[0: len(item_list)] = np.array(item_list)
                item_array = np.array(item_list)
                length.append(len(item_list))
                newtext = np.concatenate((newtext, item_array))
                # newtext.append(item_array)
            text = newtext
        results['text'] = text
        results['length'] = length

    def decode(self, results):
        """ convert text-index into text-label. """
        text_index, length = results
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.itos[t[i].item()])
            text = ' '.join(char_list)
            texts.append(text)
            index += l
        return texts

    def __call__(self, results):
        if isinstance(results, dict):
            self.encode(results)
            return results
        elif isinstance(results, tuple):
            return self.decode(results)


@register_pipeline
class GetPixelLinkMap(object):
    """
    Pixel 热点图获取
    """
    def __init__(self, config):
        self.config = config
        self.croped_size = config['croped_size']


    @staticmethod
    def points_to_contour(points):
        contours = [[list(p)] for p in points]
        return np.asarray(contours, dtype=np.int32)

    @staticmethod
    def find_contours(mask, method=None):
        if method is None:
            method = cv2.CHAIN_APPROX_SIMPLE
        mask = np.asarray(mask, dtype=np.uint8)
        mask = mask.copy()
        try:
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                           method=method)
        except:
            _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                              method=method)
        return contours

    @staticmethod
    def draw_contours(img, contours, idx=-1, color=1, border_width=1):
        #     img = img.copy()
        cv2.drawContours(img, contours, idx, color, border_width)
        return img

    @staticmethod
    def get_neighbours_8(x, y):
        """
        Get 8 neighbours of point(x, y)
        """
        return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
                (x - 1, y), (x + 1, y), \
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

    @staticmethod
    def get_neighbours_4(x, y):
        return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]

    @staticmethod
    def is_valid_cord(x, y, w, h):
        """
        Tell whether the 2D coordinate (x, y) is valid or not.
        If valid, it should be on an h x w image
        """
        return x >= 0 and x < w and y >= 0 and y < h

    @staticmethod
    def get_neighbours(x, y, neighbour_type):
        # import config
        if neighbour_type == "PIXEL_NEIGHBOUR_TYPE_4":
            return GetPixelLinkMap.get_neighbours_4(x, y)
        else:
            return GetPixelLinkMap.get_neighbours_8(x, y)

    @staticmethod
    def points_to_contours(points):
        return np.asarray([GetPixelLinkMap.points_to_contour(points)])

    @staticmethod
    def cal_gt_for_single_image(normed_xs, normed_ys, labels, config):
        """
        Args:
            xs, ys: both in shape of (N, 4),
                and N is the number of bboxes,
                their values are normalized to [0,1]
            labels: shape = (N,), only two values are allowed:
                                                            -1: ignored
                                                            1: text
        Return:
            pixel_cls_label
            pixel_cls_weight
            pixel_link_label
            pixel_link_weight
        """
        # import config
        # print(config)
        # config = GetPixelLinkMap.config
        score_map_shape = config.score_map_shape
        pixel_cls_weight_method = config.pixel_cls_weight_method
        h, w = score_map_shape
        text_label = config.text_label
        ignore_label = config.ignore_label
        background_label = config.background_label
        num_neighbours = config.num_neighbours
        bbox_border_width = config.bbox_border_width
        pixel_cls_border_weight_lambda = config.pixel_cls_border_weight_lambda
        # validate the args
        assert np.ndim(normed_xs) == 2
        assert np.shape(normed_xs)[-1] == 4
        assert np.shape(normed_xs) == np.shape(normed_ys)
        assert len(normed_xs) == len(labels)
        #     assert set(labels).issubset(set([text_label, ignore_label, background_label]))
        num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
        # rescale normalized xys to absolute values
        xs = normed_xs * w
        ys = normed_ys * h
        # initialize ground truth values
        mask = np.zeros(score_map_shape, dtype=np.int32)
        pixel_cls_label = np.ones(score_map_shape, dtype=np.int32) * background_label
        pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)
        pixel_link_label = np.zeros((h, w, num_neighbours), dtype=np.int32)
        pixel_link_weight = np.ones((h, w, num_neighbours), dtype=np.float32)
        # find overlapped pixels, and consider them as ignored in pixel_cls_weight
        # and pixels in ignored bboxes are ignored as well
        # That is to say, only the weights of not ignored pixels are set to 1
        ## get the masks of all bboxes
        bbox_masks = []
        pos_mask = mask.copy()
        for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
            if labels[bbox_idx] == background_label:
                continue
            bbox_mask = mask.copy()
            bbox_points = zip(bbox_xs, bbox_ys)
            bbox_contours = GetPixelLinkMap.points_to_contours(bbox_points)
            GetPixelLinkMap.draw_contours(bbox_mask, bbox_contours, idx=-1,
                          color=1, border_width=-1)
            bbox_masks.append(bbox_mask)
            if labels[bbox_idx] == text_label:
                pos_mask += bbox_mask
        # treat overlapped in-bbox pixels as negative,
        # and non-overlapped  ones as positive
        pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
        num_positive_pixels = np.sum(pos_mask)
        ## add all bbox_maskes, find non-overlapping pixels
        sum_mask = np.sum(bbox_masks, axis=0)
        not_overlapped_mask = sum_mask == 1
        ## gt and weight calculation
        for bbox_idx, bbox_mask in enumerate(bbox_masks):
            bbox_label = labels[bbox_idx]
            if bbox_label == ignore_label:
                # for ignored bboxes, only non-overlapped pixels are encoded as ignored
                bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
                pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
                continue
            if labels[bbox_idx] == background_label:
                continue
            # from here on, only text boxes left.
            # for positive bboxes, all pixels within it and pos_mask are positive
            bbox_positive_pixel_mask = bbox_mask * pos_mask
            # background or text is encoded into cls gt
            # print(type(bbox_positive_pixel_mask), type(bbox_label))
            pixel_cls_label += bbox_positive_pixel_mask * bbox_label
            # for the pixel cls weights, only positive pixels are set to ones
            if pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_all_ones":
                pixel_cls_weight += bbox_positive_pixel_mask
            elif pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_bbox_balanced":
                # let N denote num_positive_pixels
                # weight per pixel = N /num_positive_bboxes / n_pixels_in_bbox
                # so all pixel weights in this bbox sum to N/num_positive_bboxes
                # and all pixels weights in this image sum to N, the same
                # as setting all weights to 1
                num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
                if num_bbox_pixels > 0:
                    per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                    per_pixel_weight = per_bbox_weight / num_bbox_pixels
                    pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight
            else:
                raise (ValueError, 'pixel_cls_weight_method not supported:%s' \
                       % (pixel_cls_weight_method))
            ## calculate the labels and weights of links
            ### for all pixels in  bboxes, all links are positive at first
            bbox_point_cords = np.where(bbox_positive_pixel_mask)
            pixel_link_label[bbox_point_cords] = 1
            ## the border of bboxes might be distored because of overlapping
            ## so recalculate it, and find the border mask
            new_bbox_contours = GetPixelLinkMap.find_contours(bbox_positive_pixel_mask)
            bbox_border_mask = mask.copy()
            GetPixelLinkMap.draw_contours(bbox_border_mask, new_bbox_contours, -1,
                          color=1, border_width=bbox_border_width * 2 + 1)
            bbox_border_mask *= bbox_positive_pixel_mask
            bbox_border_cords = np.where(bbox_border_mask)
            ## give more weight to the border pixels if configured
            pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda
            ### change link labels according to their neighbour status
            border_points = zip(*bbox_border_cords)

            def in_bbox(nx, ny):
                return bbox_positive_pixel_mask[ny, nx]

            for y, x in border_points:
                neighbours = GetPixelLinkMap.get_neighbours(x, y, config.pixel_neighbour_type)
                for n_idx, (nx, ny) in enumerate(neighbours):
                    if not GetPixelLinkMap.is_valid_cord(nx, ny, w, h) or not in_bbox(nx, ny):
                        pixel_link_label[y, x, n_idx] = 0
        pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
        pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis=-1)
        heat_map = np.zeros((h, w, 10), dtype=float)
        heat_map[:, :, 0] = pixel_cls_label
        heat_map[:, :, 1] = pixel_cls_weight
        heat_map[:, :, 2:6] = pixel_link_label
        heat_map[:, :, 6:] = pixel_link_weight
        # return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight
        heat_map = heat_map.astype(np.float32)
        return heat_map

    def __call__(self, results):
        bboxes_target = results['bboxes']
        bboxes_target = np.array(bboxes_target, dtype=np.float32)
        norm_x = np.hstack((bboxes_target[:, 0].reshape(-1, 1), bboxes_target[:, 2].reshape(-1, 1),
                            bboxes_target[:, 2].reshape(-1, 1), bboxes_target[:, 0].reshape(-1, 1)))
        norm_x = np.array(norm_x, dtype=np.float32)
        norm_x /= self.croped_size[0]
        norm_y = np.hstack((bboxes_target[:, 1].reshape(-1, 1), bboxes_target[:, 1].reshape(-1, 1),
                            bboxes_target[:, 3].reshape(-1, 1), bboxes_target[:, 3].reshape(-1, 1)))
        norm_y = np.array(norm_y, dtype=np.float32)
        norm_y /= self.croped_size[1]
        labels = np.ones(norm_x.shape[0], dtype=np.int32)
        results['heat_map'] = GetPixelLinkMap.cal_gt_for_single_image(norm_x, norm_y, labels, self.config)

        return results


# @register_pipeline
# class Collect(object):
#     """
#     Collect data from the loader relevant to the specific task.
#
#     This is usually the last stage of the data loader pipeline. Typically keys
#     is set to some subset of "img", "proposals", "gt_bboxes",
#     "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
#
#     The "img_meta" item is always populated.  The contents of the "img_meta"
#     dictionary depends on "meta_keys". By default this includes:
#
#         - "img_shape": shape of the image input to the network as a tuple
#             (h, w, c).  Note that images may be zero padded on the bottom/right
#             if the batch tensor is larger than this shape.
#
#         - "scale_factor": a float indicating the preprocessing scale
#
#         - "flip": a boolean indicating if image flip transform was used
#
#         - "filename": path to the image file
#
#         - "ori_shape": original shape of the image as a tuple (h, w, c)
#
#         - "pad_shape": image shape after padding
#
#         - "img_norm_cfg": a dict of normalization information:
#             - mean - per channel mean subtraction
#             - std - per channel std divisor
#             - to_rgb - bool indicating if bgr was converted to rgb
#     """
#
#     def __init__(self,
#                  keys,
#                  meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
#                             'scale_factor', 'flip', 'img_norm_cfg')):
#         self.keys = keys
#         self.meta_keys = meta_keys
#
#     def __call__(self, results):
#         data = {}
#         img_meta = {}
#         for key in self.meta_keys:
#             img_meta[key] = results[key]
#         data['img_meta'] = DC(img_meta, cpu_only=True)
#         for key in self.keys:
#             data[key] = results[key]
#         return data
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={}, meta_keys={})'.format(
#             self.keys, self.meta_keys)
 
        
# def build_from_cfg(cfg, registry, default_args=None):
#     """Build a module from config dict.
#
#     Args:
#         cfg (dict): Config dict. It should at least contain the key "type".
#         registry (:obj:`Registry`): The registry to search the type from.
#         default_args (dict, optional): Default initialization arguments.
#
#     Returns:
#         obj: The constructed object.
#     """
#     assert isinstance(cfg, dict) and 'type' in cfg
#     assert isinstance(default_args, dict) or default_args is None
#     args = cfg.copy()
#     obj_type = args.pop('type')
#     if mmcv.is_str(obj_type):
#         obj_cls = registry.get(obj_type)
#         if obj_cls is None:
#             raise KeyError('{} is not in the {} registry'.format(
#                 obj_type, registry.name))
#     elif inspect.isclass(obj_type):
#         obj_cls = obj_type
#     else:
#         raise TypeError('type must be a str or valid type, but got {}'.format(
#             type(obj_type)))
#     if default_args is not None:
#         for name, value in default_args.items():
#             args.setdefault(name, value)
#     return obj_cls(**args)



