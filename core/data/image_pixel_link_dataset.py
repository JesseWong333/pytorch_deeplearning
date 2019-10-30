import os
import cv2
import torch.utils.data as data
import random
import numpy as np
from .utils import random_scale_cn, load_annoataion_quard, random_crop, image_normalize, load_annoataion
label_prefix = 'labels'
# import logging
# from utils.logger import setup_normal_logger
from . import register_dataset

#hcn 这一部分被合并到配置文件了
# class PixelLinkConfig():
#     score_map_shape = (128, 256)
#     stride = 4
#     pixel_cls_weight_method = "PIXEL_CLS_WEIGHT_bbox_balanced"
#     text_label = 1
#     ignore_label = -1
#     background_label = 0
#     num_neighbours = 4
#     bbox_border_width = 1
#     pixel_cls_border_weight_lambda = 1.0
#     pixel_neighbour_type = "PIXEL_NEIGHBOUR_TYPE_4"
#     decode_method = "DECODE_METHOD_join"
#     min_area = 10
#     min_height = 1
#     pixel_conf_threshold = 0.7
#     link_conf_threshold = 0.5


# config = PixelLinkConfig()


def draw_contours(img, contours, idx=-1, color=1, border_width=1):
#     img = img.copy()
    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype=np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP,
                                   method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP,
                                  method = method)
    return contours


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y, config):
    # import config
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == "PIXEL_NEIGHBOUR_TYPE_4":
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;

def cal_gt_for_single_image(normed_xs, normed_ys, labels, config):
    """
    Args:
        xs, ys: both in shape of (N, 4),
            and N is the number of bboxes,
            their values are normalized to [0,1]
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        1: text
    Return:
        pixel_cls_label
        pixel_cls_weight
        pixel_link_label
        pixel_link_weight
    """
    # import config
    # print(config)
    score_map_shape = config.score_map_shape
    pixel_cls_weight_method = config.pixel_cls_weight_method
    h, w = score_map_shape
    text_label = config.text_label
    ignore_label = config.ignore_label
    background_label = config.background_label
    num_neighbours = config.num_neighbours
    bbox_border_width = config.bbox_border_width
    pixel_cls_border_weight_lambda = config.pixel_cls_border_weight_lambda

    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)

    #     assert set(labels).issubset(set([text_label, ignore_label, background_label]))

    num_positive_bboxes = np.sum(np.asarray(labels) == text_label)
    # rescale normalized xys to absolute values
    xs = normed_xs * w
    ys = normed_ys * h

    # initialize ground truth values
    mask = np.zeros(score_map_shape, dtype=np.int32)
    pixel_cls_label = np.ones(score_map_shape, dtype=np.int32) * background_label
    pixel_cls_weight = np.zeros(score_map_shape, dtype=np.float32)

    pixel_link_label = np.zeros((h, w, num_neighbours), dtype=np.int32)
    pixel_link_weight = np.ones((h, w, num_neighbours), dtype=np.float32)

    # find overlapped pixels, and consider them as ignored in pixel_cls_weight
    # and pixels in ignored bboxes are ignored as well
    # That is to say, only the weights of not ignored pixels are set to 1

    ## get the masks of all bboxes
    bbox_masks = []
    pos_mask = mask.copy()
    for bbox_idx, (bbox_xs, bbox_ys) in enumerate(zip(xs, ys)):
        if labels[bbox_idx] == background_label:
            continue

        bbox_mask = mask.copy()

        bbox_points = zip(bbox_xs, bbox_ys)
        bbox_contours = points_to_contours(bbox_points)
        draw_contours(bbox_mask, bbox_contours, idx=-1,
                               color=1, border_width=-1)

        bbox_masks.append(bbox_mask)

        if labels[bbox_idx] == text_label:
            pos_mask += bbox_mask

    # treat overlapped in-bbox pixels as negative,
    # and non-overlapped  ones as positive
    pos_mask = np.asarray(pos_mask == 1, dtype=np.int32)
    num_positive_pixels = np.sum(pos_mask)

    ## add all bbox_maskes, find non-overlapping pixels
    sum_mask = np.sum(bbox_masks, axis=0)
    not_overlapped_mask = sum_mask == 1

    ## gt and weight calculation
    for bbox_idx, bbox_mask in enumerate(bbox_masks):
        bbox_label = labels[bbox_idx]
        if bbox_label == ignore_label:
            # for ignored bboxes, only non-overlapped pixels are encoded as ignored
            bbox_ignore_pixel_mask = bbox_mask * not_overlapped_mask
            pixel_cls_label += bbox_ignore_pixel_mask * ignore_label
            continue

        if labels[bbox_idx] == background_label:
            continue
        # from here on, only text boxes left.

        # for positive bboxes, all pixels within it and pos_mask are positive
        bbox_positive_pixel_mask = bbox_mask * pos_mask
        # background or text is encoded into cls gt
        # print(type(bbox_positive_pixel_mask), type(bbox_label))
        pixel_cls_label += bbox_positive_pixel_mask * bbox_label

        # for the pixel cls weights, only positive pixels are set to ones
        if pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_all_ones":
            pixel_cls_weight += bbox_positive_pixel_mask
        elif pixel_cls_weight_method == "PIXEL_CLS_WEIGHT_bbox_balanced":
            # let N denote num_positive_pixels
            # weight per pixel = N /num_positive_bboxes / n_pixels_in_bbox
            # so all pixel weights in this bbox sum to N/num_positive_bboxes
            # and all pixels weights in this image sum to N, the same
            # as setting all weights to 1
            num_bbox_pixels = np.sum(bbox_positive_pixel_mask)
            if num_bbox_pixels > 0:
                per_bbox_weight = num_positive_pixels * 1.0 / num_positive_bboxes
                per_pixel_weight = per_bbox_weight / num_bbox_pixels
                pixel_cls_weight += bbox_positive_pixel_mask * per_pixel_weight
        else:
            raise (ValueError, 'pixel_cls_weight_method not supported:%s' \
                              % (pixel_cls_weight_method))

        ## calculate the labels and weights of links
        ### for all pixels in  bboxes, all links are positive at first
        bbox_point_cords = np.where(bbox_positive_pixel_mask)
        pixel_link_label[bbox_point_cords] = 1

        ## the border of bboxes might be distored because of overlapping
        ## so recalculate it, and find the border mask
        new_bbox_contours = find_contours(bbox_positive_pixel_mask)
        bbox_border_mask = mask.copy()
        draw_contours(bbox_border_mask, new_bbox_contours, -1,
                               color=1, border_width=bbox_border_width * 2 + 1)
        bbox_border_mask *= bbox_positive_pixel_mask
        bbox_border_cords = np.where(bbox_border_mask)

        ## give more weight to the border pixels if configured
        pixel_cls_weight[bbox_border_cords] *= pixel_cls_border_weight_lambda

        ### change link labels according to their neighbour status
        border_points = zip(*bbox_border_cords)

        def in_bbox(nx, ny):
            return bbox_positive_pixel_mask[ny, nx]

        for y, x in border_points:
            neighbours = get_neighbours(x, y, config)
            for n_idx, (nx, ny) in enumerate(neighbours):
                if not is_valid_cord(nx, ny, w, h) or not in_bbox(nx, ny):
                    pixel_link_label[y, x, n_idx] = 0

    pixel_cls_weight = np.asarray(pixel_cls_weight, dtype=np.float32)
    pixel_link_weight *= np.expand_dims(pixel_cls_weight, axis=-1)
    heat_map = np.zeros((h, w, 10), dtype=float)
    heat_map[:, :, 0] = pixel_cls_label
    heat_map[:, :, 1] = pixel_cls_weight
    heat_map[:, :, 2:6] = pixel_link_label
    heat_map[:, :, 6:] = pixel_link_weight
    # return pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight
    return heat_map


@register_dataset
class ImagePixelLinkDataset(data.Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, dataroot, config):
        super(ImagePixelLinkDataset, self).__init__()
        self.dataroot = dataroot
        self.config = config
        self.pixel_mean = self.config.pixel_mean
        self.pixel_std = self.config.pixel_std
        img_files = []
        exts = ['png', 'jpg', 'jpeg', 'JPG']

        for parent, dirnames, filenames in os.walk(os.path.join(self.dataroot, "images")):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        img_files.append(os.path.join(parent, filename))
                        break

        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.INFO)
        # logger = setup_normal_logger(__name__)
        print('Find {} images'.format(len(img_files)))
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 如果当前的index有问题，需要继续随机的找一个
        save_path = "/media/Data/hcn/project/pytorch-ocr-framework/checkpoints/tmp_images"
        while True:
            im_fn = self.img_files[index]
            img_whole = cv2.imread(im_fn)
            # h, w, c = img_whole.shape
            # im_info = np.array([h, w, c]).reshape([1, 3])  # 3*1 变为1*3

            _, fn = os.path.split(im_fn)
            fn, _ = os.path.splitext(fn)
            txt_fn = os.path.join(self.dataroot, label_prefix, fn + '.txt')
            if not os.path.exists(txt_fn):
                # print("Ground truth for image {} not exist!".format(im_fn))
                index = random.randint(0, len(self.img_files) - 1)
                continue
            bboxs = load_annoataion(txt_fn)
            if len(bboxs) == 0:
                # print("Ground truth for image {} empty!".format(im_fn))
                index = random.randint(0, len(self.img_files) - 1)
                continue

            # 增广操作
            # 以一定的概率进行增广
            #可能公式数据并不适合，尝试去掉
            if random.random() > 0.5:
                img_whole, bboxs = random_scale_cn(img_whole, bboxs)

            # 最后我要裁剪到多大呢? 实际题块的平均尺寸， 约为 w=1500. 比较担心LSTM层的加入对长度的影响
            # 这个尺寸仍然是一个较大的尺寸， 1536 * 512  为512*512的三倍
            croped_size = (1024, 512)  # w * h
            img_croped, bboxes_target = random_crop(img_whole, bboxs, croped_size)
            img_save = img_croped.copy()
            basename = os.path.basename(im_fn)
            # for bbox in bboxes_target:
            #     cv2.rectangle(img_save, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
            #
            # cv2.imwrite(os.path.join(save_path, basename), img_save)


            if len(bboxes_target) <= 0:
                index = random.randint(0, len(self.img_files) - 1)
                continue

            # 创建对应的heat map作为label, 1/4 map_size
            # print(type(bboxes_target))
            bboxes_target = np.array(bboxes_target, dtype=np.float32)
            # norm_x = bboxes_target[:, 0::2]
            # norm_y = bboxes_target[:, 1::2]
            # print(bboxes_target[:, 0].shape, bboxes_target[:, 2].shape)
            norm_x = np.hstack((bboxes_target[:, 0].reshape(-1, 1), bboxes_target[:, 2].reshape(-1, 1), bboxes_target[:, 2].reshape(-1, 1), bboxes_target[:, 0].reshape(-1, 1)))
            norm_x = np.array(norm_x, dtype=np.float32)
            norm_x /= croped_size[0]
            norm_y = np.hstack((bboxes_target[:, 1].reshape(-1, 1), bboxes_target[:, 1].reshape(-1, 1), bboxes_target[:, 3].reshape(-1, 1), bboxes_target[:, 3].reshape(-1, 1)))

            norm_y = np.array(norm_y, dtype=np.float32)
            norm_y /= croped_size[1]
            labels = np.ones(norm_x.shape[0], dtype=np.int32)

            # print("nori", norm_y.shape, norm_x.shape, bboxes_target.shape, labels.shape)

            # pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight = \
            #     cal_gt_for_single_image(norm_x, norm_y, labels, self.config)
            heat_map = cal_gt_for_single_image(norm_x, norm_y, labels, self.config)
            img_croped_norm = image_normalize(img_croped, self.pixel_mean, self.pixel_std)
            # return img_croped_norm, pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight
            return img_croped_norm, heat_map

    @staticmethod
    def name():
        return 'ImagePixelLinkDataset'

    @staticmethod
    def collate_fn(batch):
        imgs = []
        heat_maps = []
        # pixel_cls_labels, pixel_cls_weights, pixel_link_labels, pixel_link_weights = [], [], [], []
        # for img, pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight in batch:
        #     img = img.transpose((2, 0, 1))  # 通道前置
        #     imgs.append(img[np.newaxis, :, :, :])
        #     pixel_link_label = pixel_link_label.transpose((2, 0, 1))
        #     pixel_link_weight = pixel_link_weight.transpose((2, 0, 1))
        #     pixel_cls_labels.append(pixel_cls_label[np.newaxis, :, :])
        #     pixel_cls_weights.append(pixel_cls_weight[np.newaxis, :, :])
        #     pixel_link_labels.append(pixel_link_label[np.newaxis, :, :, :])
        #     pixel_link_weights.append(pixel_link_weight[np.newaxis, :, :, :])
        #
        # return np.concatenate(imgs, axis=0), [np.concatenate(pixel_cls_labels, axis=0), \
        #                                       np.concatenate(pixel_cls_weights, axis=0), \
        #                                       np.concatenate(pixel_link_labels, axis=0), \
        #                                       np.concatenate(pixel_link_weights, axis=0)]
        #change by hcn
        for img, heat_map in batch:
            img = img.transpose((2, 0, 1))  # 通道前置
            imgs.append(img[np.newaxis, :, :, :])
            heat_maps.append(heat_map[np.newaxis, :, :])
        return np.concatenate(imgs, axis=0), np.concatenate(heat_maps, axis=0)


# if __name__ == '__main__':
#     import torch
#
#     save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'
#
#     class Opt():
#         dataroot = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/'
#     opt = Opt()
#
#     data_loader = torch.utils.data.DataLoader(
#         ImageLineDataset(opt), shuffle=False, batch_size=1, num_workers=1,
#         collate_fn=ImageLineDataset.collate_fn)
#     # 注意这样得到的通道数在最后，默认的
#     for ii, (image, heatmap) in enumerate(data_loader):
#         image = np.squeeze(image, axis=0)
#         image = image.transpose((1, 2, 0))
#         new_im = image.copy()
#         # new_im = new_im.astype(np.uint8)
#         print(ii)