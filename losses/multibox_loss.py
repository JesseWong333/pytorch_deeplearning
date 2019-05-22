import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from box_utils import match, log_sum_exp

GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def convert_coord(tri_coord, h_w_ration=0.5, variances=0.2):
    """
    改写此函数， 传入的不是[x_min, y_min, y_max]， 这里应该是传入编码后的值 [c_x, c_y, g_x], 这里写的应该是
    :tri_coord: (B*N) *3, 其中的每个元素为[x_min, y_min, y_max],
    :return: [x_min, y_min, x_max, y_max]
    """
    c_x = tri_coord[:, :, 0]
    c_y = tri_coord[:, :, 1]
    g_w = tri_coord[:, :, 2]
    # 得到g_h
    g_h = g_w
    quard_coord = torch.cat(
        (c_x.unsqueeze(-1), c_y.unsqueeze(-1), g_w.unsqueeze(-1), g_h.unsqueeze(-1)), dim=2)

    return quard_coord


# def convert_coord_head(tri_coord, h_w_ration, variances):
#     """
#     改写此函数， 传入的不是[x_min, y_min, y_max]， 这里应该是传入编码后的值 [c_x, c_y, g_x], 这里写的应该是
#     :tri_coord: (B*N) *3, 其中的每个元素为[x_min, y_min, y_max],
#     :return: [x_min, y_min, x_max, y_max]
#     """
#     c_x = tri_coord[:, :, 0]
#     c_y = tri_coord[:, :, 1]
#     g_w = tri_coord[:, :, 2]
#     # 得到g_h
#     g_h = g_w - torch.log(h_w_ration)/variances[1]
#     quard_coord = torch.cat(
#         (c_x.unsqueeze(-1), c_y.unsqueeze(-1), g_w.unsqueeze(-1), g_h.unsqueeze(-1)), dim=2)
#
#     return quard_coord


# def convert_coord_tail(tri_coord):
#     if len(tri_coord.shape) == 2:
#         x_max = tri_coord[:, 0]
#         y_min = tri_coord[:, 1]
#         y_max = tri_coord[:, 2]
#         x_min = x_max - (y_max - y_min)
#         quard_coord = torch.cat(
#             (x_min.unsqueeze(-1), y_min.unsqueeze(-1), x_max.unsqueeze(-1), y_max.unsqueeze(-1)), dim=1)
#     elif len(tri_coord.shape) == 3:
#         x_max = tri_coord[:, :, 0]
#         y_min = tri_coord[:, :, 1]
#         y_max = tri_coord[:, :, 2]
#         x_min = x_max - (y_max - y_min)
#         quard_coord = torch.cat(
#             (x_min.unsqueeze(-1), y_min.unsqueeze(-1), x_max.unsqueeze(-1), y_max.unsqueeze(-1)), dim=2)
#
#     return quard_coord


class MultiBoxHeadLoss(nn.Module):
    """
    MultiBoxHeadLoss, 这里只匹配头
    """

    def __init__(self, num_classes, overlap_thresh, neg_pos):
        super(MultiBoxHeadLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        # self.background_label = bkg_label
        # self.use_prior_for_matching = prior_for_matching
        # self.do_neg_mining = neg_mining  #  默认做negative mining
        self.negpos_ratio = neg_pos
        # self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        使用一个分类的值， 分别表示 背景 0， 头 1， 尾 2， loc坐标输出6组(前三组表示头的(xmin, ymin, ymax), (xmax, ymin, ymax))
        预测的不是这三个值！！！
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes) # 类别为两个, 头或尾，只需要用一个标量
                loc shape: torch.size(batch_size,num_priors,3)  # 最后的坐标改为输出三个
                x_min, y_min, x_max, y_max  --> xmin, ymin, ymax
                priors shape: torch.size(num_priors, 4)
            priors: 先验框 (num_priors, 4)
            targets: list类型, (num_gt, 5)  xmin, ymin, xmax, ymax   GT 要求传真实的四角坐标
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, _, conf_data = predictions
        # 将预测的部分进行坐标转换
        loc_data = convert_coord(loc_data)
        conf_data = conf_data[:, :, [0, 1]]

        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # 转换targets, 只要是头的标签
        targets_heads = []
        for idx in range(num):
            head_index = targets[idx][:, -1] == 1  # 针对头
            heads = targets[idx][head_index]
            targets_heads.append(heads)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets_heads[idx][:, :-1]
            labels = targets_heads[idx][:, -1]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)  # loc_t，和conf_t是通过传入返回的
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # conf_tmp = conf_t.cpu().numpy()
        # wrap targets
        # loc_t = Variable(loc_t, requires_grad=False)    # 编码之后的坐标
        # conf_t = Variable(conf_t, requires_grad=False)  # [batch_size, num_priors] 每个prior box的框对应标签, 最大可能性的


        pos = conf_t > 0   # 不是背景的位置

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # pos_idx对应不是背景的位置，
        loc_p = loc_data[pos_idx].view(-1,4)  # pos_idx 剩下的就都是前景的位置了， 所以这里的回归的坐标值是只考虑前景的
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining 默认做hard negative mining, 前面的选项并没有起作用, 所谓的hard mining这里就是加权
        loss_c[pos.view(-1)] = 0  # filter out pos boxes for now  # 将所有的前景loss_c 置位0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(str(conf_p.size()) + str(targets_weighted.size()))
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)  #

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        return loss_l, loss_c


class MultiBoxTailLoss(nn.Module):
    """
    MultiBoxHeadLoss, 这里只匹配头
    """

    def __init__(self, num_classes, overlap_thresh, neg_pos):
        super(MultiBoxTailLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        # self.background_label = bkg_label
        # self.use_prior_for_matching = prior_for_matching
        # self.do_neg_mining = neg_mining  #  默认做negative mining
        self.negpos_ratio = neg_pos
        # self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        使用一个分类的值， 分别表示 背景 0， 头 1， 尾 2， loc坐标输出6组(前三组表示头的(xmin, ymin, ymax), (xmax, ymin, ymax))
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes) # 类别为两个, 头或尾，只需要用一个标量
                loc shape: torch.size(batch_size,num_priors,3)  # 最后的坐标改为输出三个
                x_min, y_min, x_max, y_max  --> xmin, ymin, ymax
                priors shape: torch.size(num_priors, 4)
            priors: 先验框 (num_priors, 4)
            targets: list类型, (num_gt, 5)  xmin, ymin, xmax, ymax   GT 要求传真实的四角坐标
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        _, loc_data, conf_data = predictions
        # 将预测的部分进行坐标转换
        loc_data = convert_coord(loc_data)
        conf_data = conf_data[:, :, [0, 2]]

        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # 转换targets, 只要是头的标签
        targets_tails = []
        for idx in range(num):
            tail_index = targets[idx][:, -1] == 2  # 针对尾部
            tails = targets[idx][tail_index]
            tails[:, -1] = 1  # 还是置为1
            targets_tails.append(tails)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets_tails[idx][:, :-1].data
            labels = targets_tails[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)  # loc_t，和conf_t是通过传入返回的
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # conf_tmp = conf_t.cpu().numpy()
        # wrap targets
        # loc_t = Variable(loc_t, requires_grad=False)    # 编码之后的坐标
        # conf_t = Variable(conf_t, requires_grad=False)  # [batch_size, num_priors] 每个prior box的框对应标签, 最大可能性的


        pos = conf_t > 0   # 不是背景的位置

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # pos_idx对应不是背景的位置，
        loc_p = loc_data[pos_idx].view(-1,4)  # pos_idx 剩下的就都是前景的位置了， 所以这里的回归的坐标值是只考虑前景的
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining 默认做hard negative mining, 前面的选项并没有起作用, 所谓的hard mining这里就是加权
        loss_c[pos.view(-1)] = 0  # filter out pos boxes for now  # 将所有的前景loss_c 置位0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        #print(str(conf_p.size()) + str(targets_weighted.size()))
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)  #

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        return loss_l, loss_c


if __name__ == '__main__':
    # 测试loss编写
    criterion = MultiBoxTailLoss(2, overlap_thresh=0.5, prior_for_matching=True,
                                 bkg_label=0, neg_mining=True, neg_pos=3, neg_overlap=0.5)
    batch_size = 4
    conf = torch.randn(batch_size, 2560, 3)
    loc = torch.randn(batch_size, 2560, 3)
    predictions = (loc, loc, conf)
    priors = torch.randn(2560, 4)
    targets = []
    for i in range(batch_size):
        target = torch.randn(15, 5)
        target[:4, -1] = 1
        target[4:, -1] = 2
        targets.append(target)  # 这个是[xmin, ymin, ymax, label]
    criterion(predictions, priors, targets)
    pass
