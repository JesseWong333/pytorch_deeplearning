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
import numpy as np
import jieba
from .lang_model.de_symbol import desymbol
import re
import enchant
en_wchecker = enchant.Dict("en_US")
from collections import defaultdict, Counter


"""todo: 一个后处理class, 如何处理callable函数呢？ 比如这里的加上一个@callable?
这个类其实在encode 的时候也会用到，不应该算后处理. strLabelConverter只能算一个解码的工具类，不应该直接修改为后处理
这里写的，很多还是不够
"""


@register_post_process
class ctc_decoder(object):
    def __init__(self, vocab_path, vocab_type, subject, beam_search=False):
        self.converter = strLabelConverter(vocab_path, vocab_type)
        self.beam_search = beam_search
        self.subject = subject

    def __call__(self, preds, src_img):  # todo: 处理src_img需求，是否有必要传？
        pred_str = ''
        if self.beam_search:
            pred_str = self.decode_bs(preds)
        else:
            pred_str = self.decode_no_bs(preds)
        return pred_str
    def decode_no_bs(self, preds):
        preds = preds[0]
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        pred_str = self.converter.convert(preds.data, preds_size.data, raw=False)
        return pred_str
    def decode_bs(self, preds):
        preds = preds[0].permute(1, 0, 2)
        preds_softmax = torch.nn.functional.softmax(preds, dim=2)
        preds_softmax_array = preds_softmax.cpu().detach().numpy()
        # beam search 结果集据概率降序排序
        # 取最高结果
        return beam_search_postprocess(preds_softmax_array, self.converter, self.subject)

def beam_search_postprocess(preds_softmax_array, converter, subject):
    # beam search 结果集据概率降序排序
    # 取最高结果
    bs_candidates = prefix_beam_search(preds_softmax_array[:, 0, :], converter)[:5]

    # 利用分词离散度作为语句通顺程度判据
    # 候选集合中存在比top1 分词离散小的则选取，否则选取top1
    if len(bs_candidates) > 1:
        # 参考候选预测对符号丢失情况进行处理(包括成对符号的处理)

        # for debug
        # print('before add end symbol, ',bs_candidates)

        # 处理句尾丢失标点
        # bs_candidates = add_sym2candidate(bs_candidates)

        # for debug
        # print('after add end symbol, ',bs_candidates)

        if subject != '英语':
            bs_batch_candidate_ppl = np.array(
                [len(list(jieba.cut(desymbol(candidate), HMM=False))) for candidate in bs_candidates])
            min_ppl_id = np.argmin(bs_batch_candidate_ppl)
            if bs_batch_candidate_ppl[0] <= bs_batch_candidate_ppl[min_ppl_id]:
                min_ppl_id = 0
            # for debug
            # print(bs_candidates, bs_batch_candidate_ppl, min_ppl_id)
        else:
            #  对英语科目，选取词汇错误最少的
            err_counts = [
                np.sum([(not en_wchecker.check(word.strip())) for word in re.split(r'\W+', c) if word.strip() != ''])
                for c in bs_candidates]
            # bs_batch_candidate_err_count = [len(en_checker.set_text(candidate)) for candidate in bs_candidates]
            min_ppl_id = np.argmin(err_counts)

            # #for debug
            # sorted_ids = np.argsort(err_counts)
            # print(sorted_ids,min_ppl_id)
            # #for debug end
        return bs_candidates[min_ppl_id]

    return bs_candidates[0]

# def beam_search_postprocess(preds_softmax_array, converter, subject):
#     # beam search 结果集据概率降序排序
#     # 取最高结果
#     bs_candidates_rel = prefix_beam_search(preds_softmax_array[:, 0, :], converter, k=5)
#     bs_candidates = list(bs_candidates_rel.keys())
#     bs_candidates_prob = np.array(list(bs_candidates_rel.values()))
#
#     if pow(bs_candidates_prob[0], 1 / (len(bs_candidates[0]) if len(bs_candidates[0]) != 0 else 1)) < 0.74 \
#             and np.sum(bs_candidates_prob[:2]) < 0.5:
#         return ''
#     # 利用分词离散度作为语句通顺程度判据
#     # 候选集合中存在比top1 分词离散小的则选取，否则选取top1
#     if len(bs_candidates) > 1:
#         # 参考候选预测对符号丢失情况进行处理(包括成对符号的处理)
#
#         # 处理句首句尾丢失标点
#         bs_candidates = add_sym2candidate(bs_candidates)
#         # 处理句首序号丢失
#         bs_candidates = correct(bs_candidates)
#
#         # for debug
#         # print('after add end symbol, ',bs_candidates)
#
#         if subject != '英语':
#             ppls = np.array([len(list(jieba.cut(desymbol(candidate), HMM=True))) for candidate in bs_candidates])
#             alpha = 1.3
#             ppl_id = np.argmax(bs_candidates_prob / (alpha * (ppls - np.min(ppls)) + 1))
#             # for debug
#             # print(bs_candidates, bs_batch_candidate_ppl, min_ppl_id)
#         else:
#             #  对英语科目，选取词汇错误最少的
#             err_counts = [
#                 np.sum([(not en_wchecker.check(word.strip())) for word in re.split(r'\W+', c) if word.strip() != ''])
#                 for c in bs_candidates]
#             # bs_batch_candidate_err_count = [len(en_checker.set_text(candidate)) for candidate in bs_candidates]
#             ppl_id = np.argmin(err_counts)
#
#         return bs_candidates[ppl_id]
#
#     return bs_candidates[0]


def prefix_beam_search(ctc, converter, lm=None, k=25, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.
    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        converter （converter）: 用于字符和index的转换的词汇表，需要传入和模型一致的词汇表
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
    Retruns:
        string: The decoded CTC output.
    """

    lm = (lambda l: 1) if lm is None else lm  # 可添加语言模型，增加识别准确度
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    # alphabet = list(vocab) + [' ', '>', '%']
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc))  # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [converter.itos[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:

            # 处理停止符，我们的模型暂时不需要
            # if len(l) > 0 and l[-1] == '>':
            #     Pb[t][l] = Pb[t - 1][l]
            #     Pnb[t][l] = Pnb[t - 1][l]
            #     continue

            for c in pruned_alphabet:
                c_ix = converter.stoi[c]
                # END: STEP 2

                # STEP 3: “Extending” with a blank
                if c == '':
                    Pb[t][l] += ctc[t][0] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3

                # STEP 4: Extending with the end character
                # 最后一位是 none blank 的 extending
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                    # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    # """
                    # # 语言模型：姓名识别没有有效的语言模型，中文识别语言模型还需要做分词
                    # elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):
                    #     lm_prob = lm(l_plus.strip(' >')) ** alpha
                    #     print('l_plus: {}, lm_prob: {}'.format(l_plus, lm_prob))
                    #     Pnb[t][l_plus] += lm_prob*ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # """
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][0] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
    # END: STEP 7

    return A_prev



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
    def __init__(self, vocab_path, vocab_type, ignore_case=True):
        self._ignore_case = ignore_case
        self.itos = {}
        self.stoi = {}
        if vocab_type == 'row':
            i = 0
            with codecs.open(vocab_path, encoding="utf-8") as fr:
                # with open(convert_list_path) as fr:
                for line in fr:
                    line = line.replace("\n", "")
                    line = line.replace("\r", "")
                    self.itos[i] = line.strip("\n")
                    self.stoi[line.strip("\n")] = i
                    i += 1
        with open(vocab_path, encoding="utf-8") as f:
            for i, c in enumerate(f.read()):
                # assert c != ' '
                self.itos[i + 1] = c
                self.stoi[c] = i + 1
        self.itos[0] = ''
        self.stoi[''] = 0
        self.voc_len = len(self.stoi)

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
    def convert(self,t, length, raw=False):
        """
        没有加bs时的，ctc结果转换函数
        :param t:
        :param length:
        :param raw:
        :return:
        """
        decode_num = self.decode(t, length, raw=False)
        decode_num = decode_num.strip().split(" ")
        decode_str = ""
        for p_num in decode_num:
            if not p_num:
                decode_str += ""
            else:
                decode_str += self.itos[int(p_num)]
        return decode_str
