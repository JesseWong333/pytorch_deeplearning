# -*- coding: utf-8 -*-
"""
add by hzc
"""
import os
import os.path as osp
from . import register_evaluator
import torch
import numpy as np
import PIL
from PIL import Image, ImageFont, Image, ImageDraw


@register_evaluator
class TextRecognitionEvaluator(object):

    def __init__(self, vis_flag=False, vis_path=None):
        self.vis_flag = vis_flag
        self.vis_path = vis_path
        font_ttf = './SourceHanSans-Normal.ttf'
        self.font = ImageFont.truetype(font_ttf, 20)

    def visualize(self, filename, preds, gt):
        img = Image.open(filename)
        w, h = img.size
        vis_img = Image.new('RGB', (w, 3 * h), (255, 255, 255))
        vis_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(vis_img)
        draw.text((10, h), preds, fill=(255, 0, 0), font=self.font)
        draw.text((10, 2 * h), gt, fill=(0, 255, 0), font=self.font)
        vis_img.save(osp.join(self.vis_path, osp.basename(filename)))

    def __call__(self, model, test_loader):
        model.eval()
        if self.vis_flag and not osp.exists(self.vis_path):
            os.makedirs(self.vis_path)
        total, correct = 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.set_input(data)
                preds_str, loss = model.evaluate()
                filenames = data[-2]
                cpu_text = data[-1]
                # print(out, loss.data, cpu_text)
                for _ in range(len(preds_str)):
                    total += 1
                    pred, text, filename = preds_str[_].strip(), cpu_text[_].strip(), filenames[_]
                    pred = ''.join(pred.split())
                    text = ''.join(text.split())
                    if text == pred:
                        correct += 1
                    # print(pred, text)
                    if self.vis_flag:
                        self.visualize(filename, pred, text)
            msg = "correct / total : %d / %d, acc: %.4f" % (correct, total, correct / total)
            print(msg)
            # model.save_log(msg)
        model.train()

