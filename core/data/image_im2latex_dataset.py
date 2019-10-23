"""
Parts of the codes from https://git.iyunxiao.com/DeepVision/pytorch-ocr-framework
by hzc wt
"""


import os
import copy
import cv2
import torch
import torch.utils.data as data
import random
import numpy as np
# from functools import partial
# import math
# from losses.box_utils import jaccard
# from .utils import load_annoataion_quard, random_crop
from itertools import islice
from itertools import cycle
from torchvision import transforms
from PIL import Image
from . import register_dataset
label_prefix = 'labels'


def get_max_shape(arrays):
    """
    Args:
        images: list of arrays

    """
    shapes = list(map(lambda x: list(x.shape), arrays))
    ndim = len(arrays[0].shape)
    max_shape = []
    for d in range(ndim):
        max_shape += [max(shapes, key=lambda x: x[d])[d]]

    return max_shape


def pad_batch_images(images, max_shape=None):
    """
    Args:
        images: list of arrays
        target_shape: shape at which we want to pad

    """

    # 1. max shape
    if max_shape is None:
        max_shape = get_max_shape(images)

    # 2. apply formating
    batch_images = 255 * np.ones([len(images)] + list(max_shape))
    # batch_images = np.squeeze(batch_images, 1)
    # print(batch_images.shape, max_shape)
    for idx, img in enumerate(images):
        batch_images[idx, :, :img.shape[1], :img.shape[2]] = img

    return batch_images.astype(np.float32)


def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """Pad formulas to the max length with id_pad and adds and id_end token
    at the end of each formula

    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas

    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32

    """
    if max_len is None:
        max_len = max(list(map(lambda x: len(x), formulas)))

    batch_formulas = id_pad * np.ones([len(formulas), max_len+1],
            dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula,
                dtype=np.int32)
        batch_formulas[idx, len(formula)] = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length


def img_sort_key(ex):
    """Sort using the size of the image: (width, height)."""
    return ex[0].shape[2], ex[0].shape[1]


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """Return whether an example is an acceptable length.
    If used with a dataset as ``filter_pred``, use :func:`partial()`
    for all keyword arguments.
    Args:
        ex (torchtext.data.Example): An object with a ``src`` and ``tgt``
            property.
        use_src_len (bool): Filter based on the length of ``ex.src``.
        use_tgt_len (bool): Similar to above.
        min_src_len (int): A non-negative minimally acceptable length
            (examples of exactly this length will be included).
        min_tgt_len (int): Similar to above.
        max_src_len (int or float): A non-negative (possibly infinite)
            maximally acceptable length (examples of exactly this length
            will be included).
        max_tgt_len (int or float): Similar to above.
    """
    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def split_corpus(data, chunk_size):
    """
    Args:
        data: cycle iterator, 返回图片路径和对应的formula
        chunk_size: 一次性读取到内存中的图片张数
    Return:
        data_sort: yield 排序后的img和对应的formula
    """
    if chunk_size <= 0:
        yield data
    else:
        while True:
            shard = list(islice(data, chunk_size))  # 每次取shard size的图片进行排序
            if not shard:
                break
            data_pair = []
            # max_h, max_w = 0, 0
            for img_path, formula in shard:
                if not os.path.exists(img_path):
                    img_path = img_path.replace('media', 'home')
                img = transforms.ToTensor()(
                    Image.fromarray(cv2.imread(img_path, 0)))  # 1 * h * w
                data_pair.append((img, formula, img_path))
            data_sort = sorted(data_pair, key=img_sort_key)
            yield data_sort

@register_dataset
class ImageIm2LatexDataset(data.Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    def __init__(self, dataroot):
        super(ImageIm2LatexDataset, self).__init__()
        self.dataroot = dataroot
        data_list = []
        # if opt.isTrain:
        file_name = "all_train_random.txt"
        # else:
        #     file_name = "all_test.txt"
        with open(os.path.join(self.dataroot, file_name), encoding='utf-8') as f:
            for line in f.readlines():
                img_file, formula = line.split(" ", 1)
                data_list.append((img_file, formula))
        random.shuffle(data_list)
        print("load {} images".format(len(data_list)))
        self.data_len = len(data_list)
        self.data = cycle(data_list)
        self.chunk_size = 500  # 一次加载chunk_size数量的图片进入内存，并按照w,h的顺序排序
        self.chunk_data = iter(next(split_corpus(self.data, self.chunk_size)))
        pass

    def __len__(self):

        return self.data_len

    def __getitem__(self, index):
        err_count = 100
        while True and err_count:
            try:
                img, formula, img_name = next(self.chunk_data)
            except:
                self.chunk_data = iter(next(split_corpus(self.data, self.chunk_size)))
                img, formula, img_name = next(self.chunk_data)
            # print("img", img.shape, torch.max(img), torch.min(img), img.dtype)
            # print(img_name)
            if len(formula) == 0 or img.shape[1] > 1900 or img.shape[2] > 1900 \
                    or img.shape[1] < 20 or img.shape[2] < 20 \
                    or img.shape[1] * img.shape[2] > 600000:
                err_count -= 1
                continue
            # if self.opt.isTrain:
            return img, formula
            # else:
            #     return img, formula, img_name

    @staticmethod
    def name():
        return 'ImageIm2LatexDataset'

    @staticmethod
    def collate_fn(batch):
        images = []
        formulas = []
        filenames = []
        c = batch[-1][0].shape[0]
        max_h = max([t[0].shape[1] for t in batch])
        max_w = max([t[0].shape[2] for t in batch])
        batch_len = len(batch[0])
        if batch_len == 2:
            for item, formula in batch:
                img = np.ones((c, max_h, max_w))
                img[:, 0:item.shape[1], 0:item.shape[2]] = item
                images.append(img[np.newaxis, :, :, :])
                formulas.append(formula.strip())
        elif batch_len == 3:
            for item, formula, filename in batch:
                img = np.ones((c, max_h, max_w))
                img[:, 0:item.shape[1], 0:item.shape[2]] = item
                images.append(img[np.newaxis, :, :, :])
                formulas.append(formula.strip())
                filenames.append(filename)
        # random.shuffle(images)
        images = np.concatenate(images, axis=0)
        # formulas = np.concatenate((formulas), axis=0)
        images = images.astype(np.float32)

        # print(images.shape, np.max(images), np.min(images), images.dtype)
        if batch_len == 2:
            return images, formulas
        else:
            return images, formulas, filenames


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/media/Data/hzc/datasets/formulas/')
    opt = parser.parse_known_args()
    opt = parser.parse_args()
    dataset = ImageIm2LatexDataset(opt)
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=5,
        num_workers=1, collate_fn=dataset.collate_fn)
    for i, _ in enumerate(data_loader):
        pass
    print("sa sa sa!")
    print("su su su!")
    os._exit(0)