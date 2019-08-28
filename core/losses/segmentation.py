# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/20

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_loss


@register_loss
class SegmentationOHEMLoss(nn.Module):
    """

    """
    def __init__(self, neg_pos=3):
        super(SegmentationOHEMLoss, self).__init__()
        self.negpos_ratio = neg_pos

    def forward(self, y_true, y_pred):
        y_pred = y_pred.permute(0, 2, 3, 1)
        batch_size, h, w, channel = y_pred.shape

        pos = y_true > 0  # positive 是跟labels 同样尺寸大小的 0 - 1 tensor

        num_pos = torch.sum(pos.long(), dim=[1, 2])  # batch_size * 4

        # 分类这里要进行OHEM， 参照SSD pytorch的写法
        cls_scores = F.softmax(y_pred, dim=3)
        neg_scores = cls_scores[:, :, :, 0]  # 负样本的概率，要提取前面的

        loss_c = neg_scores.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)   # 我也不知道写的啥，但我照着写了
        _, idx_rank = loss_idx.sort(1)  # ???, 怎么还排序了???
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=loss_c.size(1) - 1)

        neg = idx_rank < num_neg.unsqueeze(1).expand_as(idx_rank)
        neg = neg.view(batch_size, h, w)

        # 正负样本均衡 1:3
        heatmap_loss = \
            self.negpos_ratio*F.cross_entropy(y_pred[pos, :], y_true[pos].long()) +\
            F.cross_entropy(y_pred[neg, :], y_true[neg].long())

        return heatmap_loss


if __name__ == '__main__':
    centre_line_loss = SegmentationOHEMLoss()
    y_true = torch.rand(2, 4, 100, 100)
    y_pred = torch.rand(2, 4, 100, 100)
    centre_line_loss(y_true, y_pred)