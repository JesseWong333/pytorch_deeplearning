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
import editdistance
import re
from hanziconv import HanziConv


def normalize_txt(st):
  """
  Normalize Chinese text strings by:
    - remove puncutations and other symbols
    - convert traditional Chinese to simplified
    - convert English chraacters to lower cases
  """
  st = ''.join(st.split(' '))
  st = re.sub("\"","",st)
  # remove any this not one of Chinese character, ascii 0-9, and ascii a-z and A-Z
  new_st = re.sub(r'[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a0-9]+','',st)
  # convert Traditional Chinese to Simplified Chinese
  new_st = HanziConv.toSimplified(new_st)
  # convert uppercase English letters to lowercase
  new_st = new_st.lower()
  return new_st


def text_distance(s1, s2):
    str1 = normalize_txt(s1)
    str2 = normalize_txt(s2)
    ed = editdistance.eval(str1, str2)
    l = max(len(str1), len(str2))
    ned = ed / l if l > 0 else 0
    return ed, ned


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
        total, correct, total_ed, total_ned = 0, 0, 0.0, 0.0
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
                    ed, ned = text_distance(text, pred)
                    total_ed += ed
                    total_ned += ned
                    if text == pred:
                        correct += 1
                    # print(pred, text)
                    if self.vis_flag:
                        self.visualize(filename, pred, text)
            msg = "correct / total : %d / %d, acc: %.4f, avg_ed: %.4f, normalized_ed: %.4f" % \
                  (correct, total, correct / total, total_ed / total, 1 - total_ned / total)
            print(msg)
            # model.save_log(msg)
        model.train()

