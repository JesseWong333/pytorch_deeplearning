"""
模型推理前的image预处理
"""
import numpy as np
import cv2
import torchvision.transforms.functional as F
from . import register_pre_process
from torchvision import transforms
from PIL import Image

@register_pre_process
def c2td_preprocess(img):
    h, w, _ = img.shape
    # 检测填充
    img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
    return img

@register_pre_process
def reg_preprocess(img):
    """
    识别模型要求高度为32的灰度图，且需要进行输入图片的归一化
    :param img:
    :return:
    """
    nh = 32
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    nw = int(nh * w / h)
    gray_img = cv2.resize(gray_img, (nw, nh))
    gray_img = np.expand_dims(gray_img, axis=2)
    gray_img = F.to_tensor(gray_img)  # 这个标准步骤我以后还是做一下比较好，进行数据归一化
    gray_img.sub_(0.5).div_(0.5)
    return gray_img

@register_pre_process
def im2latex_preprocess(image):
    images = []
    formulas = []
    image = image_resize(image, 600000)
    # print(image.shape)
    image = transforms.ToTensor()(Image.fromarray(image))
    img = np.ones((image.shape[0], image.shape[1], image.shape[2]))
    img[:, 0:image.shape[1], 0:image.shape[2]] = image
    images.append(img[np.newaxis, :, :, :])
    images = np.concatenate(images, axis=0)
    images = images.astype(np.float32)
    formulas.append('')
    input_data = (images, formulas)
    return input_data



def image_resize(image, maxpixels):
    """
    识别模型，图片预处理
    图片大小超过设定的阈值，先缩小，防止送入网络的图片过大
    :param image:
    :return:
    """
    h, w = image.shape[:2]
    if w * h >= maxpixels:  # the shape of image is too large
        nw = int(w / 4 * 3)
        nh = int(h / 4 * 3)
        while nw * nh >= maxpixels:
            nw = int(w / 4 * 3)
            nh = int(h / 4 * 3)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return image


@register_pre_process
def pixellink_preprocess(img):
    h, w, _ = img.shape
    img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
    img = image_normalize(img, pixel_mean, std_mean)
    return img


pixel_mean = [0.91610104, 0.91612387, 0.91603917]
std_mean = [0.2469206, 0.24691878, 0.24692364]
def image_normalize(img, pixel_mean, std):
    img = img.astype(np.float32)
    img /= 255
    h, w, c = img.shape
    for i in range(c):
        img[:, :, i] -= pixel_mean[i]
        img[:, :, i] /= std[i]
    return img

