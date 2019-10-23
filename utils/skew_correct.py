# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/10/17
"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by mcm
"""
import cv2
import numpy as np
from scipy.stats import entropy
def rotate_bound(image, rotate_angle, pad_value=(0, 0, 0)):
    """
    旋转图片，超出原图大小的边界用指定颜色填充.
    :param image: 待旋转图片
    :param rotate_angle:  旋转角度
    :param pad_value: 边界填充颜色
    :return:
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -rotate_angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=pad_value)


def correct_image(src_image):
    """
    倾斜校正.
    :param src_image: 待校正图片
    :return:
    """
    if src_image.ndim == 3:
        gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = src_image

    _, bin_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    thetas = np.linspace(-2, 2., 50, endpoint=False)
    entropys = []
    for angle in thetas:
        rotated_image = rotate_bound(bin_image, angle)

        sum_row = np.sum(rotated_image, axis=1)
        if not float(np.sum(sum_row)):
            continue
        sum_row = sum_row / float(np.sum(sum_row))
        entropys.append(entropy(sum_row))

    #try:
    #    angle = thetas[np.argmin(entropys)]
    #except Exception:
    #    return src_image, 0
    if not entropys:
        return src_image, 0
    else:
        angle = thetas[np.argmin(entropys)]

    # (-00, -0.2)U(0.2, +00)范围内旋转, [0, 0.2]不旋转
    if abs(angle) > 0.2:
        ret_image = rotate_bound(src_image, angle, (255, 255, 255))
        return ret_image, angle

    return src_image, angle


if __name__ == "__main__":
    import glob
    import os

    def show_image(show_name, show_img, scale=0.5):
        show_img = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow(show_name, show_img)
        cv2.waitKey(0)

    res_dir = "./data/res"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    im_path_list = glob.glob("./data/sknew_data/*.png")
    num_of_imgs = len(im_path_list)
    for idx, im_path in enumerate(im_path_list, 1):
        print("{}/{}, {}".format(idx, num_of_imgs, im_path))
        img = cv2.imread(im_path, 0)

        res_img, _ = correct_image(img)

        basename = os.path.basename(im_path)
        out_path = os.path.join(res_dir, basename)
        cv2.imwrite(out_path, res_img)