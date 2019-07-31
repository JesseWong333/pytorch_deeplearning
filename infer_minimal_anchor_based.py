# 仅供测试使用， 不要加入git

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import numpy as np
from post_process import post_process
from utils.preprocess_util import correct_image
from core.networks import VGGRFBNet


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img, angle = correct_image(img)
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


if __name__ == '__main__':

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'

    device = torch.device('cuda')
    # device = 'cpu'

    net = VGGRFBNet(in_channels=3, channel_width=1.0, norm_type='batch')

    state_dict = torch.load('/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/anchor_based/195_net_net.pth')
    # new_state_dict = OrderedDict()

    net.load_state_dict(state_dict)

    net = net.to(device)

    img_gen = img_generator('/media/Data/wangjunjie_code/advancedEast/dataset/junior_biology')

    from tqdm import tqdm
    for img, file_name in tqdm(img_gen):
        h, w, _ = img.shape
        img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        h, w, _ = img.shape

        img_t = img.transpose((2, 0, 1)).astype(np.float32)  # 通道前置

        img_t = torch.from_numpy(img_t)
        img_t = img_t.unsqueeze(dim=0).to(device)

        # forward pass
        net_out = net(img_t)
        # conf = net_out[2].detach().numpy()
        c_dets_head, c_dets_tail = post_process(net_out, h, w)

        for i in range(len(c_dets_head)):
            cv2.rectangle(img, (c_dets_head[i][0], c_dets_head[i][1]), (c_dets_head[i][2], c_dets_head[i][3]),
                          color=[0, 0, 255], thickness=2)

        for i in range(len(c_dets_tail)):
            cv2.rectangle(img, (c_dets_tail[i][0], c_dets_tail[i][1]), (c_dets_tail[i][2], c_dets_tail[i][3]),
                          color=[0, 255, 0], thickness=2)
        cv2.imwrite(os.path.join(save_path, file_name + '.png'), img)

