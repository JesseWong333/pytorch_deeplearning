# 仅供测试使用， 不要加入git

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import numpy as np
from utils.preprocess_util import correct_image
from post_process.centre_line_process import process
from models.networks import VGGPixel


def process_bboxes(bboxes, h, w, slide_thresh=0.3):
    """
    对结果再做一些预处理。包括 边缘的滤除，边框的缩放，
    :param bboxes:
    :return:
    """
    # 要考虑是否为0的情形
    bboxes = [( max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])) for bbox in bboxes]

    bboxes = set(bboxes)
    # slide_bboxes = set()
    # for bbox in bboxes:
    #     xmin, ymin, xmax, ymax = bbox
    #     box_h = ymax - ymin
    #     thresh_h = slide_thresh * box_h
    #     if ymin < thresh_h or (h - ymax) < thresh_h:
    #         slide_bboxes.add(bbox)
    #
    # # 计算平均高度不能包括边界。碰到了边界上的一堆点变成了一连串的碎掉的点.
    #
    # centre_bboxes = bboxes - slide_bboxes
    # if len(centre_bboxes) > 0:
    #     bboxs_h = [(bbox[3] - bbox[1]) for bbox in centre_bboxes]
    # else:
    #     bboxs_h = [(bbox[3] - bbox[1]) for bbox in bboxes]
    # bboxs_h.sort()  # inplace sort
    # median = bboxs_h[int(len(bboxs_h) / 2)]  # 一张图片文字高度的中位数
    #
    # for bbox in bboxes.copy():
    #     if bbox[3] - bbox[1] > median * 1.5 or bbox[3] - bbox[1] < median * 0.5:
    #         bboxes.remove(bbox)
    #
    # # 进一步去判断slide_bboxes
    # for bbox in slide_bboxes:
    #     if bbox[3] - bbox[1] < median * 0.7:
    #         if bbox in bboxes:  # 可能前面已经删除过了
    #             bboxes.remove(bbox)

    # 上边界的比例， 下边界的比例. 宽度为32像素的图，实际文字大小只有24. 上下只扩充实际大小的1/6
    # bboxes_p = set()
    # for bbox in bboxes:
    #     expand = (bbox[3] - bbox[1])/6
    #     bboxes_p.add((bbox[0], max(0, bbox[1]-expand),  bbox[2], min(h, bbox[3]+expand)))
    # 统一4像素扩充
    bboxes_p = set()
    for bbox in bboxes:
        # expand = (bbox[3] - bbox[1]) / 6
        bboxes_p.add((bbox[0], max(0, bbox[1] - 4), bbox[2], min(h, bbox[3] + 4)))
    return bboxes_p


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'jpg', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img, angle = correct_image(img)
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


if __name__ == '__main__':

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'

    device = torch.device('cuda')
    # device = 'cpu'S

    net = VGGPixel(in_channels=3, out_channel=3, channel_width=1.0, norm_type='batch')

    state_dict = torch.load('/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/pixel_based/100_net_net.pth')
    # new_state_dict = OrderedDict()

    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()

    # ori_imgs_path = "/media/Data/wangjunjie_code/advancedEast/dataset/junior_biology"
    ori_imgs_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo/'
    img_gen = img_generator(ori_imgs_path)

    from tqdm import tqdm
    for img, file_name in tqdm(img_gen):
        h, w, _ = img.shape
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        # h, w, _ = img.shape

        img_t = img.transpose((2, 0, 1)).astype(np.float32)  # 通道前置

        img_t = torch.from_numpy(img_t)
        img_t = img_t.unsqueeze(dim=0).to(device)

        # forward pass
        with torch.no_grad():
            net_out = net(img_t)
        cords = process(net_out, img, file_name)

        cords = process_bboxes(cords, h, w)

        for cord in cords:
            xmin, ymin, xmax, ymax = map(int, cord)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          color=[0, 0, 255], thickness=2)
        cv2.imwrite(os.path.join(save_path, file_name + '.png'), img)

        with open(os.path.join(save_path, file_name + '.txt'), 'w') as f_txt:
            for cord in cords:
                f_txt.write(','.join(map(str, cord)) + '\n')
