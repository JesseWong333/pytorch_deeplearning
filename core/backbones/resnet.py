"""
Parts of the codes form https://git.iyunxiao.com/DeepVision/text-detection-ctpn/blob/deploy_CT2D/main_server.py
by wt
"""

import collections
import Levenshtein
import cv2
import numpy as np
import torch
import torch.nn as nn

import queue
from . import register_backbone

from PIL import Image, ImageDraw, ImageFont



def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    text = " " if text == "" else text
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "NotoSansCJK-Black.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def update_names(pred, names):
    if pred not in names:
        return
    names[pred] -= 1
    if names[pred] == 0:
        names.pop(pred)  # 剔除已查找过的名字


def min_l_norm(dst, names):
    """
    对每一个可能的预测值，查找编辑距离最小的目标预测值
    :param dst: prob_pred
    :param names: 目标空间
    :return: (l_norm_distance, pred)
    """
    l_norm_distances = list(map(lambda x: (Levenshtein.distance(dst, x) / max(len(dst), len(x)), x), names))
    return min(l_norm_distances)


def count_l_distance(l_distances, final_preds, names):
    """
    对每一个可能预测值计算最小归一化编辑距离，取最小值后，对剩下的重新计算最小编辑距离
    :param l_distances: {(prob_pred, gt, img_path, _): (l_norm_distance, final_pred)}
    :return: final_preds: ((selected pred, gt, img_path, _), final pred)
    """
    while len(l_distances) != 0:
        min_l_distance = min(l_distances.items(), key=lambda item: item[1][0])
        final_preds.append((min_l_distance[0], min_l_distance[1][1]))
        update_names(min_l_distance[1][1], names)
        l_distances = {k: l_distances[k] for k in l_distances if k[2] != min_l_distance[0][2]}
        for k in l_distances:
            if l_distances[k][1] not in names:
                l_distances[k] = min_l_norm(k[0], names)
    return


def match_total(sim_beam_serachs, gt_infos, names):
    n_correct = 0
    n_correct_old = 0
    err = []
    que = queue.Queue()
    [que.put(z) for z in zip(sim_beam_serachs, gt_infos)]
    preds_bak = []
    total = len(gt_infos)
    while not que.empty():
        sims, gt_info = que.get()
        if total > 0:
            if sims[0] == gt_info[0]:
                n_correct_old += 1
            total -= 1

        if sims[0] == '_':
            n_correct += 1
            update_names(gt_info[0], names)
            continue
        if sims[0] in names:
            update_names(sims[0], names)
            if sims[0] == gt_info[0]:
                n_correct += 1
            else:
                err.append((gt_info[1], sims[0], sims[0], gt_info[0]))
        else:
            sims = sims[1:]
            if sims == []:
                index = gt_infos.index((gt_info))
                preds_bak.append((sim_beam_serachs[index], gt_infos[index]))
            else:
                que.put((sims, gt_info))
    return n_correct, n_correct_old, preds_bak, err


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
            0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
            1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
            2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
            3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
            3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


class BidirectionalResLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, nOut, dropout):
        super(BidirectionalResLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.embedding = nn.Linear(rnn_size * 2, nOut)

        for _ in range(num_layers):
            self.layers.append(nn.LSTM(input_size, rnn_size, bidirectional=True))
            # input_size = rnn_size

    def forward(self, input_feed):
        # h_0, c_0 = hidden
        h_1, c_1 = [], []
        pre_input = torch.zeros(input_feed.shape).to(input_feed.device)
        for i, layer in enumerate(self.layers):
            layer.flatten_parameters()
            h_1_i, c_1_i = layer(input_feed + pre_input)
            pre_input = input_feed.clone()
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)

        T, b, h = h_1_i.size()
        t_rec = h_1_i.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


@register_backbone
class ResBilstm(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.res_net = ResNet(1, 512, BasicBlock, [1, 2, 5, 3])
        self.rnn = BidirectionalResLSTM(3, 512, 256, self.n_class, 0.2)

    def forward(self, x):
        x = self.res_net(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        x = x.permute(1, 0, 2)
        return x

