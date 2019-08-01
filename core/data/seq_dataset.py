# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/31

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import cv2
from . import register_dataset


"""
这个比较特殊的是，需要选择适用什么alignCollate
"""


@register_dataset
class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            if self.transform is not None:
                img = self.transform(img)
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode())
            if self.target_transform is not None:
                label = self.target_transform(label)
        return img, label


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.ANTIALIAS):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index
        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        # newimages=[]
        # for image in images:
        #     transform = resizeNormalize((image.size[0],image.size[1]))
        #     newimages.append(transform(image))
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels

# added by Jesse Wong 在折叠的时候，不做任何处理
class  alignCollate_no_fill(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.transform = resizeNormalize_new((800, 32))  # the size is not used

    def __call__(self, batch):
        # 只将高度缩放到32，其余部分不做处理
        ori_images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        images = []
        # 缩放
        for image in ori_images:
            # image = Image.open(imgpath).convert('L')
            w, h = image.size
            nw = imgH * w / h
            if nw % 4 == 0:
                nw = int(nw)
            else:
                nw = int(int(nw / 4 + 1) * 4)
            image = image.resize((nw, imgH), Image.ANTIALIAS)
            images.append(image)

        new_images = [self. transform(image) for image in images]
        return new_images, labels


def border_fill(img,top=2, buttom=2, left=5, right=0):
    """
    图像左边像素填充
    :param img:
    :param pixelnum:
    :return:
    """
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.copyMakeBorder(img, top, buttom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    finalimg = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return finalimg

#add by hcn
class alignCollate_new(object):
    #输入的图片需要是已经做过宽高统一处理的图片
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
    def __call__(self, batch):
        #这里从dataset里面获取到的是图片的路径和标注，是自由加载数据的方式
        imgpaths, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        images = []
        for imgpath in imgpaths:
            image = Image.open(imgpath).convert('L')
            images.append(image)
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        # newimages=[]
        # for image in images:
        #     transform = resizeNormalize((image.size[0],image.size[1]))
        #     newimages.append(transform(image))
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


class alignCollate_randomw(object):
    #输入的同一batch图片的宽度不一致，找出最大的宽度，将图片都统一到同一宽高，不足的填充空白
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
    def __call__(self, batch):
        ori_images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        images = []
        new_images = []
        maxw = 0
        for image in ori_images:
            # image = Image.open(imgpath).convert('L')
            w, h = image.size
            nw = imgH * w / h
            if nw % 4 == 0:
                nw = int(nw)
            else:
                nw = int(int(nw / 4 + 1) * 4)
            image = image.resize((nw, imgH), Image.ANTIALIAS)
            images.append(image)
            if nw > maxw:
                maxw = nw
        for image in images:
            w, h = image.size
            if w < maxw:
                image = image.convert('RGB')
                image = border_fill(image, 0, 0, 0, maxw - w)
            new_images.append(image.convert('L'))
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
        transform = resizeNormalize_new((maxw, imgH))
        new_images = [transform(image) for image in new_images]
        new_images = torch.cat([t.unsqueeze(0) for t in new_images], 0)
        return new_images, labels, maxw, imgH


class resizeNormalize_new(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img