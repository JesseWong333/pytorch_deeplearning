"""
处理使用centre line的方式的结果
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from . import register_post_process


def draw_activate(im, activation_pixels, file_name, pixel_size=4, save_path='/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'):
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


def crop(img, bboxes):
    """

    :param img: 3通道的图片
    :param bboxes:
    :return:
    """
    h, w, _ = img.shape
    bboxes = [(max(0, bbox[0]), max(0, bbox[1] - 4), min(w, bbox[2]), min(h, bbox[3] + 4)) for bbox in bboxes]
    bboxes = [tuple(map(int, bbox)) for bbox in bboxes]
    img_patches = []
    for bbox in bboxes:
        img_patch = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
        img_patches.append(img_patch)
    return img_patches


@register_post_process
def c2td_process_horizon(outputs, img, file_name=None, thresh=0.8, save_path='/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'):
    """

    :param outputs: the output values from model. A dict.
    :return:
    """
    out = outputs[0]  # 重构得不好，为什么单独写'pred'
    out = out.permute(0, 2, 3, 1).squeeze().cpu().numpy()  # h*w*3
    scores = out[:, :, 0]

    scores_ori = scores.copy()

    scores[scores > 1.] = 1.
    scores[scores < 0] = 0.
    # 转换为 [0~255]
    scores = scores*255

    # todo: 调试代码： 可视化激活
    # cond = np.greater_equal(scores, 255*thresh)
    # activation_pixels = np.where(cond)
    # draw_activate(Image.fromarray(img), activation_pixels, file_name)

    # 或者这里可以先计算图片的梯度
    # _, bin = cv2.threshold(scores, 125, 255, cv2.THRESH_BINARY)
    # edges = cv2.Canny(scores.astype(np.uint8), 50, 150, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1,  np.pi / 180, 10, minLineLength=10, maxLineGap=1)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]*4
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite(os.path.join(save_path, file_name + '_line.png'), img)

    _, bin = cv2.threshold(scores, 255*thresh, 255, cv2.THRESH_BINARY)

    # 为什么有的版本是返回是二个，有的是三个?
    contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = list(filter(contour_filter, contours))

    # todo: 调试代码
    # contours_4 = [c.copy()*4 for c in contours]
    # img_contours = cv2.drawContours(img.copy(), contours_4, -1, (0, 0, 255), 2)  # 这个要乘以四
    # cv2.imwrite(os.path.join(save_path, file_name + '_contours.png'), img_contours)

    # img_rectangle = img.copy()

    cords = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # todo 调试代码
        # cv2.rectangle(img_rectangle, (x*4, y*4), ((x+w)*4, (y+h)*4),
        #               color=[0, 0, 255], thickness=2)

        # aspect_ratio = float(w) / h  # 不应该看矩形的宽高比， 应该按照最后的宽高比滤掉
        # if aspect_ratio < 1:
        #     continue
        xmin = (x-1.5) * 4   # 原来都是设置0.5
        xmax = (x+w+1.5) * 4

        roi = scores_ori[y:y+h, x:x+w]
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
            ave_top = top/len(regress_pixel[0])
            ave_down = down/len(regress_pixel[1])
            cords.append((xmin, ave_top, xmax, ave_down))

    # 滤除噪音
    cords = list(filter(lambda cord: (cord[2] - cord[0])/(cord[3]-cord[1]+1) > 1.8, cords))


    # todo 调试代码
    # cv2.imwrite(os.path.join(save_path, file_name + '_rectangle.png'), img_rectangle)
    img_patches = crop(img, cords)
    return img_patches