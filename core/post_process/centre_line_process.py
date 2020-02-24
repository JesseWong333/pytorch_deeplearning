"""
处理使用centre line的方式的结果
code from wjj
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from . import register_post_process

def draw_activate(im, activation_pixels, file_name, pixel_size=4, save_path='/home/chen/hcn/data/pixel_link_data/ocr_szt_test_ckpt/senior_math_res/'):
    draw_act = ImageDraw.Draw(im)
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        px = (j + 0.5) * pixel_size
        py = (i + 0.5) * pixel_size
        line_width, line_color = 1, 'red'
        draw_act.line([(px - 0.5 * pixel_size, py - 0.5 * pixel_size),
                       (px + 0.5 * pixel_size, py - 0.5 * pixel_size),
                       (px + 0.5 * pixel_size, py + 0.5 * pixel_size),
                       (px - 0.5 * pixel_size, py + 0.5 * pixel_size),
                       (px - 0.5 * pixel_size, py - 0.5 * pixel_size)],
                      width=line_width, fill=line_color)

    im.save(os.path.join(save_path, file_name + '_act.png'))


def contour_filter(x):
    isnoise = cv2.contourArea(x) > 4  # 有的在算面积的时候这里滤除掉了, why, 有的明明是有一行的, 改在rectangle滤除
    return isnoise


def process_bboxes(bboxes, ratio, img_shape):
    """
    对检测框再做一些后处理。包括 边框超出边界的处理，边缘的滤除，边框的缩放，
    :param bboxes:
    :return:
    """
    # 要考虑是否为0的情形
    h, w, _ = img_shape
    bboxes = [(max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])) for bbox in bboxes]
    bboxes = set(bboxes)
    # 上边界的比例， 下边界的比例. 宽度为32像素的图，实际文字大小只有24. 上下只扩充实际大小的1/6
    bboxes_p = set()
    for bbox in bboxes:
        expand = 4
        bboxes_p.add((bbox[0]/ratio, max(0, bbox[1]/ratio - expand), bbox[2]/ratio, min(h, bbox[3]/ratio + expand)))
        # bboxes_p.add((bbox[0] / ratio, bbox[1] / ratio, bbox[2] / ratio, bbox[3] / ratio))
    bboxes_p = np.array(list(bboxes_p), dtype=int)
    return bboxes_p

@register_post_process
class centre_line_process(object):

    def __init__(self, thresh=0.8):
        self.thresh = thresh

    def __call__(self, out, img, ratio, src_img_shape):
        """
           :param out: 1*3*h*w
           :return:
           """
        out = out[0]
        out = out.permute(0, 2, 3, 1).squeeze().cpu().numpy()  # h*w*3
        scores = out[:, :, 0]

        scores_ori = scores.copy()

        scores[scores > 1.] = 1.
        scores[scores < 0] = 0.
        # 转换为 [0~255]
        scores = scores * 255

        # todo: 调试代码： 可视化激活
        cond = np.greater_equal(scores, 255 * self.thresh)

        _, bin = cv2.threshold(scores, 255 * self.thresh, 255, cv2.THRESH_BINARY)

        # 为什么有的版本是返回是二个，有的是三个?
        try:
            contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            _, contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        # todo: 调试代码
        contours_4 = [c.copy() * 4 for c in contours]

        img_rectangle = img.copy()

        cords = []
        center_cords = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # todo 调试代码
            cv2.rectangle(img_rectangle, (x * 4, y * 4), ((x + w) * 4, (y + h) * 4),
                          color=[0, 0, 255], thickness=2)

            # aspect_ratio = float(w) / h  # 不应该看矩形的宽高比， 应该按照最后的宽高比滤掉
            # if aspect_ratio < 1:
            #     continue
            xmin = (x - 0.5) * 4
            xmax = (x + w + 0.5) * 4
            # add by hcn
            ymin = (y - 0.5) * 4
            ymax = (y + h + 0.5) * 4
            center_cords.append((xmin, ymin, xmax, ymax))
            # add by hcn

            roi = scores_ori[y:y + h, x:x + w]
            # 只看其中的部分像素的高和宽即可
            cond = np.greater_equal(roi, 0.9)
            regress_pixel = np.where(cond)
            if len(regress_pixel[0]) == 0:
                cond = np.greater_equal(roi, 0.8)  # 再降低一次
                regress_pixel = np.where(cond)
            top = 0
            down = 0
            for i, j in zip(regress_pixel[0], regress_pixel[1]):
                j = j + x
                i = i + y
                py = (i + 0.5) * 4
                top += py + out[i, j, 1]
                down += py + out[i, j, 2]

            if len(regress_pixel[0]) > 0:
                ave_top = top / len(regress_pixel[0])
                ave_down = down / len(regress_pixel[1])
                cords.append((xmin, ave_top, xmax, ave_down))
                # 根据左右边界回归结果 计算左右边界

        # 滤除噪音
        cords = list(
            filter(lambda cord: (cord[2] - cord[0]) / (cord[3] - cord[1] + 1) > 1 and (cord[3] - cord[1]) > 0, cords))
        # cords = list(filter(lambda cord: (cord[3] - cord[1]) / (cord[2] - cord[0] + 1) > 2.5, cords))
        cords = process_bboxes(cords, ratio, src_img_shape)

        # todo 调试代码
        # cv2.imwrite(os.path.join(save_path, file_name + '_rectangle.png'), img_rectangle)

        return cords


# @register_post_process
# def centre_line_process(out, img, ratio, src_img_shape, thresh=0.8):
#     """
#     :param out: 1*3*h*w
#     :return:
#     """
#     out = out[0]
#     out = out.permute(0, 2, 3, 1).squeeze().cpu().numpy()  # h*w*3
#     scores = out[:, :, 0]
#
#     scores_ori = scores.copy()
#
#     scores[scores > 1.] = 1.
#     scores[scores < 0] = 0.
#     # 转换为 [0~255]
#     scores = scores*255
#
#     # todo: 调试代码： 可视化激活
#     cond = np.greater_equal(scores, 255*thresh)
#
#     _, bin = cv2.threshold(scores, 255*thresh, 255, cv2.THRESH_BINARY)
#
#     # 为什么有的版本是返回是二个，有的是三个?
#     _, contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     # contours = list(filter(contour_filter, contours))
#
#     # todo: 调试代码
#     contours_4 = [c.copy()*4 for c in contours]
#
#     img_rectangle = img.copy()
#
#     cords = []
#     center_cords = []
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#
#
#
#         # todo 调试代码
#         cv2.rectangle(img_rectangle, (x*4, y*4), ((x+w)*4, (y+h)*4),
#                       color=[0, 0, 255], thickness=2)
#
#         # aspect_ratio = float(w) / h  # 不应该看矩形的宽高比， 应该按照最后的宽高比滤掉
#         # if aspect_ratio < 1:
#         #     continue
#         xmin = (x-0.5) * 4
#         xmax = (x+w+0.5) * 4
#         # add by hcn
#         ymin = (y-0.5) * 4
#         ymax = (y+h+0.5) * 4
#         center_cords.append((xmin, ymin, xmax, ymax))
#         # add by hcn
#
#         roi = scores_ori[y:y+h, x:x+w]
#         # 只看其中的部分像素的高和宽即可
#         cond = np.greater_equal(roi, 0.9)
#         regress_pixel = np.where(cond)
#         if len(regress_pixel[0]) == 0:
#             cond = np.greater_equal(roi, 0.8)  # 再降低一次
#             regress_pixel = np.where(cond)
#         top = 0
#         down = 0
#         for i, j in zip(regress_pixel[0], regress_pixel[1]):
#             j = j + x
#             i = i + y
#             py = (i + 0.5) * 4
#             top += py + out[i, j, 1]
#             down += py + out[i, j, 2]
#
#         if len(regress_pixel[0]) > 0:
#             ave_top = top/len(regress_pixel[0])
#             ave_down = down/len(regress_pixel[1])
#             cords.append((xmin, ave_top, xmax, ave_down))
#             # 根据左右边界回归结果 计算左右边界
#
#     # 滤除噪音
#     cords = list(filter(lambda cord: (cord[2] - cord[0])/(cord[3]-cord[1]+1) > 1 and (cord[3] - cord[1]) > 0, cords))
#     # cords = list(filter(lambda cord: (cord[3] - cord[1]) / (cord[2] - cord[0] + 1) > 2.5, cords))
#     cords = process_bboxes(cords, ratio, src_img_shape)
#
#
#     # todo 调试代码
#     # cv2.imwrite(os.path.join(save_path, file_name + '_rectangle.png'), img_rectangle)
#
#     return cords