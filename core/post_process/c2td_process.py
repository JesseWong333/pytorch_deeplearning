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
    return img_patches, bboxes


@register_post_process
def c2td_process_horizon(outputs, img, file_name=None, thresh=0.8, save_path='/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'):
    """

    :param outputs: the output values from model. A dict.
    :return:
    """
    out = outputs[0]  # 可能有多个输出，是一起送入到这里进行后处理的
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
    img_patches, cords = crop(img, cords)
    return img_patches, cords


def convert_cord(point, out):
    x, y = point
    py = (y + 0.5) * 4
    top = py + out[y, x, 1]
    down = py + out[y, x, 2]
    return top, down


def cal_height(top_point, bottom_point, out):
    # 计算真实的高
    top, _ = convert_cord(top_point, out)
    _, down = convert_cord(bottom_point, out)
    return down - top + 1


def process_horizon(cnt, out, scores_ori):
    x, y, w, h = cv2.boundingRect(cnt)
    xmin = (x - 1.5) * 4  # expand the left and right by around 4 pixels
    xmax = (x + w + 1.5) * 4
    roi = scores_ori[y:y + h, x:x + w]
    cond = np.greater_equal(roi, 0.9)
    regress_pixel = np.where(cond)
    if len(regress_pixel[0]) == 0:
        cond = np.greater_equal(roi, 0.8)
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
        if (xmax - xmin) / (ave_down - ave_top + 1e-4) > 1.8:  # todo 过滤阈值，需考虑合理性
            return xmin, ave_top-4, xmax, ave_down+4  # expand the top and bottom by 4 pixles
        else:
            return None
    else:
        return None


def expand_curved_boxes(cords, expand_pixels=4):
    xmin, y_left_top = cords[0]
    _, y_left_down = cords[1]
    xmax, y_right_top = cords[-2]
    _, y_right_down = cords[-1]
    cords.insert(0, [xmin - expand_pixels, y_left_down])
    cords.insert(0, [xmin - expand_pixels, y_left_top])
    cords.append([xmax + int(expand_pixels*2), y_right_top])
    cords.append([xmax + int(expand_pixels*2), y_right_down])
    return cords


def draw_cross(img, point, color=(0, 0, 255), size=8, thickness=1):
    cv2.line(img, (point[0]-int(size/2), point[1]), (point[0]+int(size/2), point[1]), color, thickness)
    cv2.line(img, (point[0], point[1]-int(size/2)), (point[0], point[1]+int(size/2)), color, thickness)


def convert_relative_cords(cords, xmin, ymin):
    cords[0] = cords[0] - xmin
    cords[2] = cords[2] - xmin
    cords[1] = cords[1] - ymin
    cords[3] = cords[3] - ymin
    return cords


def get_cords_slide(cord_1, cord_2):
    xmin = min(cord_1[0], cord_1[2])
    ymin = min(cord_1[1], cord_2[1])
    xmax = max(cord_2[0], cord_2[2])
    ymax = max(cord_1[3], cord_2[3])
    return xmin, ymin, xmax, ymax


def affine_deform(cords, img):
    cords = np.array(cords)
    h, w, _ = img.shape
    # clip the cords that are not within the img
    cords[cords < 0] = 0
    cords[cords[:, 0] > w, 0] = w
    cords[cords[:, 1] > h, 1] = h
    xmin = np.min(cords[:, 0])
    xmax = np.max(cords[:, 0])  # max是可以取到的
    ymin = np.min(cords[:, 1])
    ymax = np.max(cords[:, 1])
    cords[:, 0] = cords[:, 0] - xmin
    cords[:, 1] = cords[:, 1] - ymin
    roi = img[ymin:ymax + 1, xmin:xmax + 1, :]
    # ------^-----^------^----------- 已经得到了相对的roi的坐标
    cords = cords.reshape(-1, 4)
    cords_dst = cords.copy()
    ave_height = np.median(cords[:, 3] - cords[:, 1]) + 1
    cords_dst[:, 1] = 0
    cords_dst[:, 3] = ave_height - 1
    # 按照距离去计算
    for i in range(cords.shape[0] - 1):
        # 利用cords计算距离， 在cords_dst中去设置
        cur_top = cords[i, 0:2]
        cur_down = cords[i, 2:4]
        next_top = cords[i + 1, 0:2]
        next_down = cords[i + 1, 2:4]
        # 算平均的距离
        dis_top = np.sqrt(np.sum(np.square(cur_top - next_top)))
        dis_down = np.sqrt(np.sum(np.square(cur_down - next_down)))
        dis = round((dis_top + dis_down) / 2)
        cords_dst[i + 1, (0, 2)] = cords_dst[i, 0] + dis
    # 分段进行affine_transform
    # 原坐标cords, 现坐标cords_dst, 计算纺射变换
    dst_patchs = []
    for i in range(cords.shape[0] - 1):
        xmin_src, ymin_src, xmax_src, ymax_src = get_cords_slide(cords[i], cords[i+1])
        roi_patch = roi[ymin_src: ymax_src+1, xmin_src:xmax_src+1, :]
        # h, w, _ = roi_patch.shape
        # if h <= 0 or w <= 0:
        #     return None  # 我并没有去调查对于某些图片，为什么这里会出现异常? 是因为坐标有负数造成的
        cur = cords[i].copy()
        nex = cords[i+1].copy()
        cur = convert_relative_cords(cur, xmin_src, ymin_src)
        nex = convert_relative_cords(nex, xmin_src, ymin_src)
        cur_top = cur[0:2]
        cur_down = cur[2:4]
        next_top = nex[0:2]
        # 代码重复， 写在一起
        xmin_dst, ymin_dst, xmax_dst, ymax_dst = get_cords_slide(cords_dst[i], cords_dst[i + 1])
        cur_dst = cords_dst[i].copy()
        nex_dst = cords_dst[i+1].copy()
        cur_dst = convert_relative_cords(cur_dst, xmin_dst, ymin_dst)
        nex_dst = convert_relative_cords(nex_dst, xmin_dst, ymin_dst)
        cur_top_dst = cur_dst[0:2]
        cur_down_dst = cur_dst[2:4]
        next_top_dst = nex_dst[0:2]
        pts1 = np.concatenate((cur_top, cur_down, next_top)).reshape(3, 2).astype(np.float32)
        pts2 = np.concatenate((cur_top_dst, cur_down_dst, next_top_dst)).reshape(3, 2).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(roi_patch, M, (xmax_dst-xmin_dst, ymax_dst-ymin_dst))
        dst_patchs.append(dst)
    img = np.concatenate(dst_patchs, axis=1)
    return img


@register_post_process
def c2td_process(outputs, img, file_name=None, thresh=0.8, sample_ratio=1, save_path='/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'):
    out = outputs[0]  # 可能有多个输出，是一起送入到这里进行后处理的
    out = out.permute(0, 2, 3, 1).squeeze().cpu().numpy()  # h*w*3

    scores = out[:, :, 0]
    scores_ori = scores.copy()
    # convert scores to [0, 255]
    scores[scores > 1.] = 1.
    scores[scores < 0] = 0.
    scores = scores * 255

    # todo: 调试代码： 可视化激活
    # cond = np.greater_equal(scores, 255 * thresh)
    # activation_pixels = np.where(cond)
    # draw_activate(Image.fromarray(img), activation_pixels, file_name)
    _, bin = cv2.threshold(scores, 255 * thresh, 255, cv2.THRESH_BINARY)
    _,contours, hierarchy = cv2.findContours(bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    curved_boxes = []
    rectangle_boxes = []
    for cnt in contours:
        # cnt: n_point*1*2
        cnt = np.squeeze(cnt, axis=1)
        leftmost = cnt[cnt[:, 0].argmin()][0]
        rightmost = cnt[cnt[:, 0].argmax()][0]
        top_point = cnt[cnt[:, 1].argmin()]
        topmost = top_point[1]
        bottom_point = cnt[cnt[:, 1].argmax()]
        bottommost = bottom_point[1]
        # len_x = rightmost - leftmost + 1
        # whether the text is needed to be corrected?
        # choose an appropriate sample ratio for the text line
        height = bottommost - topmost + 1
        step = int(height * sample_ratio)
        # convert the height to real height
        real_height = cal_height(top_point, bottom_point, out)
        sampled_x = list(range(leftmost, rightmost + 1, step))
        if len(sampled_x) < 3:
            # the text line is too short, we do not perform deform. Save this to a list
            cords = process_horizon(cnt, out, scores_ori)
            if cords is not None:
                rectangle_boxes.append(cords)
            continue
        # add the the last point at the edge
        if rightmost - sampled_x[-1] > 0:
            sampled_x.append(rightmost)
        # calculate ave_sample_height
        sampled_x_dict = {x: [] for x in sampled_x}
        for i in range(0, len(cnt)):  # cnt只是边缘的点
            cur_x = cnt[i, 0]
            cur_y = cnt[i, 1]
            if sampled_x_dict.get(cur_x) is not None:
                sampled_x_dict.get(cur_x).append(cur_y)
        # inplace sort
        for key in sampled_x_dict.keys():
            sampled_x_dict[key].sort()
        ave_sample_height = 0
        for x in sampled_x_dict.keys():
            min_y = sampled_x_dict[x][0]
            max_y = sampled_x_dict[x][-1]
            sample_height = cal_height((x, min_y), (x, max_y), out)
            ave_sample_height += sample_height
        ave_sample_height = ave_sample_height / len(sampled_x_dict)
        if (real_height / ave_sample_height + 1e-4) < 1.2:  # todo 阈值，判断是否需要校正，需考虑合理性. 越小的值，校正的越多
            cords = process_horizon(cnt, out, scores_ori)
            if cords is not None:
                rectangle_boxes.append(cords)
            continue
        cords = []  # 一行的N个坐标点
        for x in sampled_x_dict.keys():
            min_y = sampled_x_dict[x][0]
            max_y = sampled_x_dict[x][-1]
            top = 0
            down = 0
            for y in range(min_y, max_y + 1):
                py = (y + 0.5) * 4
                top += py + out[y, x, 1]
                down += py + out[y, x, 2]
            ave_top = top / (max_y - min_y + 1)
            ave_down = down / (max_y - min_y + 1)
            cords.append([int((x + 0.5) * 4), int(ave_top - 4)])
            cords.append([int((x + 0.5) * 4), int(ave_down + 4)])
        # expand left and right by 4 pixel  一共要扩充四个点。 如果后面坐了横向的text_moutain 说不定就不需要了
        cords = expand_curved_boxes(cords)
        curved_boxes.append(cords)
    # todo 调试代码 画关键点
    # img_keypoints = img.copy()
    # for boxes in curved_boxes:
    #     for point in boxes:
    #         draw_cross(img_keypoints, point)
    # cv2.imwrite(os.path.join(save_path, file_name + '_keypoint.png'), img_keypoints)

    img_patches, rectangle_boxes = crop(img, rectangle_boxes)

    correct_patches = []
    for ii, cords in enumerate(curved_boxes):
        line = affine_deform(cords, img)
        correct_patches.append(line)
    return img_patches + correct_patches, rectangle_boxes + curved_boxes