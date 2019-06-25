import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEML1Loss(nn.Module):

    def __init__(self, neg_pos=2):
        super(OHEML1Loss, self).__init__()
        self.negpos_ratio = neg_pos
        # self.l1Loss = nn.SmoothL1Loss()

    def forward(self, y_pred, y_true):

        batch_size, h, w, _ = y_true.shape
        y_pred = y_pred.view(batch_size, -1)
        y_true = y_true.view(batch_size, -1)

        pos = y_true != 0.  # positive 是跟labels 同样尺寸大小的 0 - 1 tensor
        # negative = labels < self.thresh

        num_pos = torch.sum(pos.long(), dim=[1]).unsqueeze(1)  # batch_size * 1

        # 分类这里要进行OHEM， 参照SSD pytorch的写法
        loss_c = torch.abs(y_pred-y_true)
        loss_c[pos] = 0.
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)   # 我也不知道写的啥，但我照着写了
        _, idx_rank = loss_idx.sort(1)  # ???, 怎么还排序了???
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=loss_c.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # neg = neg.view(batch_size, h, w)

        # 正负样本均衡 1:3
        heatmap_loss = \
            self.negpos_ratio*F.smooth_l1_loss(y_pred[pos], y_true[pos]) +\
            F.smooth_l1_loss(y_pred[neg], y_true[neg])

        return heatmap_loss


if __name__ == '__main__':
    centre_line_loss = CentreLineLoss(0.3)
    y_true = torch.rand(1, 100, 100, 3)
    y_pred = torch.rand(1, 100, 100, 3)
    centre_line_loss(y_true, y_pred)
