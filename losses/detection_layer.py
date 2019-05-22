import torch
import math
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.autograd import Variable
from box_utils import decode, nms


# def convert_coord_head(tri_coord, h, w):
#     """
#     将一个三点的头坐标，转换为四点的坐标
#     :tri_coord: (B*N) *3, 其中的每个元素为[x_min, y_min, y_max],
#     :return: [x_min, y_min, x_max, y_max]
#     """
#
#     x_min = tri_coord[:, :, 0]
#     y_min = tri_coord[:, :, 1]
#     y_max = tri_coord[:, :, 2]
#     x_max = x_min + (y_max - y_min) * w / h
#     quard_coord = torch.cat(
#         (x_min.unsqueeze(-1), y_min.unsqueeze(-1), x_max.unsqueeze(-1), y_max.unsqueeze(-1)), dim=2)
#
#     return quard_coord
#
#
# def convert_coord_tail(tri_coord, h, w):
#     x_max = tri_coord[:, :, 0]
#     y_min = tri_coord[:, :, 1]
#     y_max = tri_coord[:, :, 2]
#     x_min = x_max - (y_max - y_min) * w / h
#     quard_coord = torch.cat(
#         (x_min.unsqueeze(-1), y_min.unsqueeze(-1), x_max.unsqueeze(-1), y_max.unsqueeze(-1)), dim=2)
#     return quard_coord


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
    g_h = g_w  # 直接相等就好了， 原来应该是 w * s_k_x * e^(g_w * v) = h * s_k_y * e^(g_h *v)
    quard_coord = torch.cat(
        (c_x.unsqueeze(-1), c_y.unsqueeze(-1), g_w.unsqueeze(-1), g_h.unsqueeze(-1)), dim=2)

    return quard_coord


class DetectHead(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    variance在创建prior的时候并没有用到
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label
        #self.thresh = thresh

        # Parameters used in nms.
        self.variance = cfg['variance']

    def forward(self, predictions, prior, h, w):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, _, conf = predictions
        conf = conf[:, :, [0, 1]]

        # todo: 这里是存在问题的，预测的编码并不是（xmin, ymin, xmax）三边坐标， 而是 中心编码x. y, 高和宽的log编码
        loc = convert_coord(loc)

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)


        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            boxes = torch.zeros(1, num_priors, 4)
            scores = torch.zeros(1, num_priors, self.num_classes)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            boxes = torch.zeros(num, num_priors, 4)
            scores = torch.zeros(num, num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''

            boxes[i] = decoded_boxes
            scores[i] = conf_scores

        return boxes, scores


class DetectTail(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    variance在创建prior的时候并没有用到
    """
    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label
        #self.thresh = thresh

        # Parameters used in nms.
        self.variance = cfg['variance']

    def forward(self, predictions, prior, h, w):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        _, loc, conf = predictions
        conf = conf[:, :, [0, 2]]
        loc = convert_coord(loc)  # 这里还是编码呐， 似乎不能这样转？？？有按照长和宽的相对,

        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)


        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            boxes = torch.zeros(1, num_priors, 4)
            scores = torch.zeros(1, num_priors, self.num_classes)

        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes)
            boxes = torch.zeros(num, num_priors, 4)
            scores = torch.zeros(num, num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            c_mask = conf_scores.gt(self.thresh)
            decoded_boxes = decoded_boxes[c_mask]
            conf_scores = conf_scores[c_mask]
            '''

            boxes[i] = decoded_boxes
            scores[i] = conf_scores

        return boxes, scores
