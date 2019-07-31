# 仅供测试使用， 不要加入git

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from utils.tps_grid_gen import TPSGridGen
from core.networks import VGGSkew


def img_generator(folder):
    file_list = [os.path.join(folder, item) for item in os.listdir(folder)]
    img_paths = list(filter(lambda item: item.split('.')[-1] == 'png', file_list))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        file_name = img_path.split('/')[-1].split('.')[0]
        yield img, file_name


def norm_fiducial_point(point, w=1024, h=512):
    point = point.reshape(-1, 2)
    cen_w = w/2
    cen_h = h/2
    point[:, 0] = (point[:, 0] - cen_w)/cen_w
    point[:, 1] = (point[:, 1] - cen_h)/cen_h
    return point


def save_tensor_img(img_t, file_name):
    img = img_t.detach().cpu().numpy().squeeze().astype(np.int)
    img = img.transpose((1, 2, 0))
    cv2.imwrite(file_name + '.png', img)


def norm_fiducial_point_t(point, fiducial_point_src, w=1024, h=512):
    point = point.permute(0, 2, 3, 1).reshape(-1, 8*16, 2)
    point = point + torch.tensor(fiducial_point_src).reshape(8*16, 2).unsqueeze(dim=0)
    cen_w = w/2
    cen_h = h/2
    point[:, 0] = (point[:, 0] - cen_w)/cen_w
    point[:, 1] = (point[:, 1] - cen_h)/cen_h
    return point


def post_process():
    pass


if __name__ == '__main__':
    # TPS moudle
    fiducial_point_x = np.arange(32, 1024 - 32 + 1, 64).astype(np.float32)  # 一共有16个点
    fiducial_point_y = np.arange(32, 512 - 32 + 1, 64).astype(np.float32)  # 共有8个，   16*8 = 128个点基点
    xx, yy = np.meshgrid(fiducial_point_x, fiducial_point_y)
    fiducial_point_src = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]), axis=-1)
    fiducial_point_src_norm = norm_fiducial_point(fiducial_point_src.copy())
    tps = TPSGridGen(512, 1024, torch.Tensor(fiducial_point_src_norm))

    save_path = '/media/Data/wangjunjie_code/pytorch_text_detection/demo_results/'

    device = torch.device('cuda')
    # device = 'cpu'S

    net = VGGSkew(in_channels=3, channel_width=1, norm_type='batch')

    state_dict = torch.load('/media/Data/wangjunjie_code/pytorch_text_detection/checkpoints/skew_model/latest_net_net.pth')
    # new_state_dict = OrderedDict()

    net.load_state_dict(state_dict)

    net = net.to(device)
    net.eval()

    ori_imgs_path = '/media/Data/wangjunjie_code/pytorch_text_detection/datasets/visualise'
    img_gen = img_generator(ori_imgs_path)

    from tqdm import tqdm
    for img, file_name in tqdm(img_gen):
        h, w, _ = img.shape
        # img = np.pad(img, ((0, 16 - h % 16), (0, 16 - w % 16), (0, 0)), 'constant', constant_values=255)  # 填充到16的倍数
        # h, w, _ = img.shape

        img_t = img.transpose((2, 0, 1)).astype(np.float32)  # 通道前置

        img_t = torch.from_numpy(img_t)
        img_t = img_t.unsqueeze(dim=0).to(device)

        # forward pass
        with torch.no_grad():
            net_out = net(img_t)
        fiducial_point_dst = norm_fiducial_point_t(net_out.detach().cpu(), fiducial_point_src)

        source_coordinate = tps(fiducial_point_dst)  # 目标是可以unsqueeze
        trans_grid = source_coordinate.view(1, 512, 1024, 2)

        # 求取一个逆变换
        step_w = 2 / 1024
        step_h = 2 / 512

        x = torch.arange(start=-1, end=1, step=step_w)
        y = torch.arange(start=-1, end=1, step=step_h)
        grid_x, grid_y = torch.meshgrid(x, y)
        refer_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1).permute(1, 0, 2).unsqueeze(0)
        reverse_trans_grid = 2 * refer_grid - trans_grid
        reverse_t = F.grid_sample(img_t.detach().cpu(), reverse_trans_grid, padding_mode='border')
        save_tensor_img(reverse_t, os.path.join(save_path, file_name))
