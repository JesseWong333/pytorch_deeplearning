import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_loss

def ohem_single(score, n_pos, neg_mask):
    if n_pos == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = neg_mask
        return selected_mask

    neg_num = neg_mask.view(-1).sum()
    neg_num = (min(n_pos * 3, neg_num)).to(torch.int)

    if neg_num == 0:
        selected_mask = neg_mask
        return selected_mask

    neg_score = torch.masked_select(score, neg_mask) * -1
    value, _ = neg_score.topk(neg_num)
    threshold = value[-1]

    selected_mask = neg_mask * (score <= -threshold)
    return selected_mask


def ohem_batch(neg_conf, pos_mask, neg_mask):
    selected_masks = []
    for img_neg_conf, img_pos_mask, img_neg_mask in zip(neg_conf, pos_mask, neg_mask):
        n_pos = img_pos_mask.view(-1).sum()
        selected_masks.append(ohem_single(img_neg_conf, n_pos, img_neg_mask))

    selected_masks = torch.stack(selected_masks, 0).to(torch.float)

    return selected_masks

@register_loss
class PixelLinkLoss(nn.Module):
    """
    预测中心线， 并且在中心线上回归
    中心线使用山峰的方式
    """
    def __init__(self):
        super(PixelLinkLoss, self).__init__()
        # self.l1Loss = nn.SmoothL1Loss()
        # self.softmax_layer = nn.Softmax2d()
        # self.pixel_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)
        # self.link_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)


    def forward(self, target,logits):
        """
        定义输出为： heat map, 使用text_map 定义， 既然是使用.  热点图使用 SmoothL1Loss
        回归的上下坐标也使用上下坐标点

        第一维， heat_map 概率， 第二维，上坐标， 第三维， 下面的坐标， 是不是很简单？
        哪些需要去预测真实的标签？ 是y_true中的
        :param y_true: [batch_size, h, w, (score, top, bottom)]
        :param y_pred:
        :return:
        """
        pixel_cls_labels = target[:, :, :, 0]
        pixel_cls_weights = target[:, :, :, 1]
        pixel_link_labels = target[:, :, :, 2:6]
        pixel_link_weights = target[:, :, :, 6:]
        n, c, h, w = logits.shape
        num_neighbours = int((c - 2) / 2)
        pixel_cls_logits = logits[:, 0:2, :, :]
        pixel_link_logits = logits[:, 2:, :, :]
        # pixel_cls_scores = self.softmax_layer(pixel_cls_logits)
        # pixel_cls_logits_flatten = pixel_cls_logits.view(n, 2, -1)
        # pixel_cls_scores_flatten = pixel_cls_scores.view(n, 2, -1)
        # pixel_link_logits = pixel_link_logits.view(n, )
        pos_mask = (pixel_cls_labels > 0)
        neg_mask = (pixel_cls_labels == 0)
        pixel_cls_loss = F.cross_entropy(pixel_cls_logits, pos_mask.to(torch.long), reduce=False)
        pixel_cls_scores = F.softmax(pixel_cls_logits, dim=1)
        pixel_neg_scores = pixel_cls_scores[:, 0, :, :]
        selected_neg_pixel_mask = ohem_batch(pixel_neg_scores, pos_mask, neg_mask)

        n_pos = pos_mask.view(-1).sum()
        n_neg = selected_neg_pixel_mask.view(-1).sum()

        pixel_cls_weights = (pixel_cls_weights + selected_neg_pixel_mask).to(torch.float)

        cls_loss = (pixel_cls_loss * pixel_cls_weights).view(-1).sum() / (n_pos + n_neg)

        # for link loss
        if n_pos == 0:
            link_loss = (pixel_link_logits * 0).view(-1).sum()
            shape = pixel_link_logits.shape
            pixel_link_logits_flat = pixel_link_logits.contiguous().view(shape[0], 2, num_neighbours, shape[2], shape[3])
        else:
            shape = pixel_link_logits.shape
            pixel_link_logits_flat = pixel_link_logits.contiguous().view(shape[0], 2, num_neighbours, shape[2], shape[3])
            link_label_flat = pixel_link_labels.permute(0, 3, 1, 2)

            pixel_link_loss = F.cross_entropy(pixel_link_logits_flat, link_label_flat.to(torch.long), reduce=False)

            def get_loss(label):
                link_mask = (link_label_flat == label)
                link_weight_mask = pixel_link_weights.permute(0, 3, 1, 2) * link_mask.to(torch.float)
                n_links = link_weight_mask.view(-1).sum()
                loss = (pixel_link_loss * link_weight_mask).view(-1).sum() / n_links
                return loss

            neg_loss = get_loss(0)
            pos_loss = get_loss(1)

            neg_lambda = 1.0
            link_loss = pos_loss + neg_loss * neg_lambda

        # loss_text = criterion(texts, gt_texts, selected_masks)
        # loss = 2 * cls_loss + 1 * link_loss



        # labels = y_true[:, :, :, 0]
        #
        # logits = y_pred[:, :, :, 0]
        #
        # positive = labels >= self.thresh  # positive 是跟labels 同样尺寸大小的 0 - 1 tensor
        # negative = labels < self.thresh
        #
        # beta = torch.mean(positive.float())
        # # 正负样本均衡 1:3
        # # print("label", labels.device, logits.device)
        # heatmap_loss = \
        #     (1.0 - beta) * self.negpos_ratio * F.smooth_l1_loss(labels[positive], logits[positive]) +\
        #     beta * F.smooth_l1_loss(labels[negative], logits[negative])
        #
        # # 对于其中的正样本要预测上下的坐标
        # cord_true = y_true[:, :, :, 1:]
        # cord_pred = y_pred[:, :, :, 1:]
        # pos_idx = positive.unsqueeze(positive.dim()).expand_as(cord_true)  # 将最后一维扩充， 再进行
        # location_loss = F.smooth_l1_loss(cord_true[pos_idx], cord_pred[pos_idx])
        return cls_loss * 2, link_loss