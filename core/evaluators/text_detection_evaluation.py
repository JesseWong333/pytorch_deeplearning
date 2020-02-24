# -*- coding: utf-8 -*-
"""
add by hzc
"""
import os
import os.path as osp
from . import register_evaluator
import torch
import numpy as np
import shapely
import PIL
import cv2
from PIL import Image, ImageFont, Image, ImageDraw


# todo: 文本检测评估器待完善
@register_evaluator
class TextDetectionEvaluator(object):
    """
    暂时仅针对((xmin, ymin, xmax, ymax))
    """

    def __init__(self, vis_flag=False, vis_path=None):
        self.vis_flag = vis_flag
        self.vis_path = vis_path
        self.Iou_thres = 0.5
        # font_ttf = './SourceHanSans-Normal.ttf'
        # self.font = ImageFont.truetype(font_ttf, 20)

    def visualize(self, filename, dt_bboxes, gt_bboxes):
        img = cv2.imread(filename)
        for bbox in dt_bboxes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=2)
        for bbox in gt_bboxes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=2)
        cv2.imwrite(osp.join(self.vis_path, osp.basename(filename)), img)

    @staticmethod
    def box_area(box):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (xmin, ymin, xmax, ymax) coordinates.
        :param box:
        :return:
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def box_iou(box1, box2):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        :param box1:
        :param box2:
        :return:
        """
        area1 = TextDetectionEvaluator.box_area(box1)
        area2 = TextDetectionEvaluator.box_area(box2)
        x_lt = max(box1[0], box1[0])
        y_lt = max(box1[1], box1[1])
        x_rb = min(box1[2], box1[2])
        y_rb = min(box1[3], box1[3])
        h = y_rb - y_lt if y_rb - y_lt > 0 else 0
        w = x_rb - x_lt if x_rb - x_lt > 0 else 0
        inter = h * w
        iou = inter / (area1 + area2 - inter)
        return iou

    def __call__(self, model, test_loader):
        model.eval()
        if self.vis_flag and not osp.exists(self.vis_path):
            os.makedirs(self.vis_path)
        all_dt_match = []
        n_gt = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.set_input(data)
                dt_bboxes, loss = model.evaluate()
                filenames = data[-1]
                gt_bboxes = data[-2]
                for _ in range(len(dt_bboxes)):
                    dt_bbox, gt_bbox, filename = dt_bboxes[_], gt_bboxes[_], filenames[_]
                    n_gt += len(gt_bboxes)
                    dt_match = []
                    if self.vis_flag:
                        self.visualize(filename, gt_bbox, dt_bbox)
                    for dt_poly in dt_bbox:
                        match = False
                        for gt_poly in gt_bbox:
                            if self.box_iou(dt_poly, gt_poly) >= self.Iou_thres:
                                match = True
                                break
                        dt_match.append(match)
                    all_dt_match.extend(dt_match)
            all_dt_match = np.array(all_dt_match, dtype=np.bool).astype(np.int)
            n_pos = np.sum(all_dt_match)
            n_dt = len(all_dt_match)
            precision = n_pos / n_dt
            recall = n_pos / float(n_gt)
            f_score = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
            msg = "f_score: %.4fd, precision: %.4f, recall: %.4f" % (f_score, precision, recall)
            print(msg)
            model.save_log(msg)
        model.train()

