import torch
import torch.nn as nn
import torch.nn.functional as F


class CentreLineLoss(nn.Module):
    """
    预测中心线， 并且在中心线上回归
    中心线使用山峰的方式
    """
    def __init__(self, thresh=0.5, neg_pos=3):
        super(CentreLineLoss, self).__init__()
        self.thresh = thresh
        self.negpos_ratio = neg_pos
        # self.l1Loss = nn.SmoothL1Loss()

    def forward(self, y_true, y_pred, OHEM=True):
        """
        定义输出为： heat map, 使用text_map 定义， 既然是使用.  热点图使用 SmoothL1Loss
        回归的上下坐标也使用上下坐标点

        第一维， heat_map 概率， 第二维，上坐标， 第三维， 下面的坐标， 是不是很简单？
        哪些需要去预测真实的标签？ 是y_true中的
        :param y_true: [batch_size, h, w, (score, top, bottom)]
        :param y_pred:
        :return:
        """
        y_pred = y_pred.permute(0, 2, 3, 1)
        batch_size, h, w, _ = y_true.shape

        labels = y_true[:, :, :, 0]
        logits = y_pred[:, :, :, 0]

        pos = labels >= self.thresh  # positive 是跟labels 同样尺寸大小的 0 - 1 tensor
        # negative = labels < self.thresh

        num_pos = torch.sum(pos.long(), dim=[1, 2]).unsqueeze(1)  # batch_size * 1

        # 分类这里要进行OHEM， 参照SSD pytorch的写法
        loss_c = torch.abs(labels-logits)
        loss_c[pos] = 0.
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)   # 我也不知道写的啥，但我照着写了
        _, idx_rank = loss_idx.sort(1)  # ???, 怎么还排序了???
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=loss_c.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = neg.view(batch_size, h, w)

        # 正负样本均衡 1:3
        heatmap_loss = \
            self.negpos_ratio*F.smooth_l1_loss(labels[pos], logits[pos]) +\
            F.smooth_l1_loss(labels[neg], logits[neg])

        # 对于其中的正样本要预测上下的坐标
        cord_true = y_true[:, :, :, 1:]
        cord_pred = y_pred[:, :, :, 1:]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(cord_true)  # 将最后一维扩充， 再进行
        location_loss = F.smooth_l1_loss(cord_true[pos_idx], cord_pred[pos_idx])
        return heatmap_loss, location_loss


if __name__ == '__main__':
    centre_line_loss = CentreLineLoss(0.3)
    y_true = torch.rand(1, 100, 100, 3)
    y_pred = torch.rand(1, 100, 100, 3)
    centre_line_loss(y_true, y_pred)
