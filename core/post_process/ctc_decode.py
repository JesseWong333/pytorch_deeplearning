# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/2

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import codecs
from . import register_post_process


"""todo: 一个后处理class, 如何处理callable函数呢？ 比如这里的加上一个@callable?
这个类其实在encode 的时候也会用到，不应该算后处理. strLabelConverter只能算一个解码的工具类，不应该直接修改为后处理
这里写的，很多还是不够
"""


@register_post_process
class ctc_decoder(object):
    def __init__(self, convert_list_path):
        convert_list = []
        with codecs.open(convert_list_path, encoding="utf-8") as fr:
            # with open(convert_list_path) as fr:
            for line in fr:
                line = line.replace("\n", "")
                line = line.replace("\r", "")
                # convert_list.append(line.decode("utf-8"))
                convert_list.append(line.strip("\n"))
        self.convert_list = convert_list
        self.converter = strLabelConverter()

    def __call__(self, preds, src_img):  # todo: 处理src_img需求，是否有必要传？
        preds = preds[0]
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        pred_num = self.converter.decode(preds.data, preds_size.data, raw=False)
        pred_num = pred_num.strip().split(" ")
        pred_str = ""
        for p_num in pred_num:
            if not p_num:
                pred_str += ""
            else:
                pred_str += self.convert_list[int(p_num)]
        return pred_str


"""
The decode method is from the original crnn_pytorch repo, and modified by hcn
"""


class strLabelConverter(object):
    """ 当初这样写一个很重要的原因就是，存储的lmdb标签本身已经转换为数字
    Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """
    def __init__(self, ):
        pass

    def encode(self, text):
        """
        Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            textsplit = text.strip().split(' ')
            text = [int(item) for item in textsplit]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = []
            newtext = []
            for item in text:
                itemsplit = item.strip().split(' ')
                length.append(len(itemsplit))
                for num in itemsplit:
                    newtext.append(int(num))
            text = newtext
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                rawtext = ''
                for i in t:
                    rawtext = rawtext + str(int(i)) + '-'
                return rawtext
            else:
                simtext = ''
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        simtext = simtext + str(int(t[i])) + ' '
                return simtext.strip()
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
