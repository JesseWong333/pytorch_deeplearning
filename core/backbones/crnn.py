# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/7/31
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_backbone
from ..modules import BidirectionalLSTM


@register_backbone
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        return output


class GRCL(nn.Module):
    def __init__(self, nc, n_out, n_iter, ks, ss, ps):
        super(GRCL, self).__init__()
        self.conv_rec_nn = nn.Conv2d(n_out, n_out, ks, ss, ps)
        self.conv_gate_rec_nn = nn.Conv2d(n_out, n_out, 1, 1, 0)
        self.bn_f_nn = nn.BatchNorm2d(n_out)
        self.conv_f_nn = nn.Conv2d(nc, n_out, ks, ss, ps)
        self.relu_0_nn = nn.ReLU(True)
        self.conv_gate_f_nn = nn.Conv2d(nc, n_out, ks, ss, ps)
        self.bn_gate_f_nn = nn.BatchNorm2d(n_out)
        self.bn_rec_nn = nn.BatchNorm2d(n_out)
        self.bn_gate_rec_nn = nn.BatchNorm2d(n_out)
        self.bn_gate_mul_nn = nn.BatchNorm2d(n_out)
        self.relu_i_nn = nn.ReLU(True)
        self.n_iter = n_iter

    def forward(self, input):
        for i in range(self.n_iter):
            if i == 0:
                # Feed forward
                conv_f = self.conv_f_nn(input)
                bn_f = self.bn_f_nn(conv_f)
                x = self.relu_0_nn(bn_f)

                # Gated
                conv_gate_f = self.conv_gate_f_nn(input)
                bn_gate_f = self.bn_gate_f_nn(conv_gate_f)
            else:
                c_rec = self.conv_rec_nn(x)
                bn_rec = self.bn_rec_nn(c_rec)

                c_gate_rec = self.conv_gate_rec_nn(x)
                bn_gate_rec = self.bn_gate_rec_nn(c_gate_rec)
                gate_add = torch.add(bn_gate_rec, bn_gate_f)
                gate = F.sigmoid(gate_add)

                gate_mul = torch.mul(bn_rec, gate)
                bn_gate_mul = self.bn_gate_mul_nn(gate_mul)
                x_add = torch.add(bn_f, bn_gate_mul)

                x = self.relu_i_nn(x_add)
        return x


@register_backbone
class GRCNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, ):
        super(GRCNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1_nn = nn.Conv2d(nc, 64, 3, 1, 1)
        self.act1_nn = nn.ReLU(True)
        self.pool1_nn = nn.MaxPool2d(2, 2)
        self.grcl1_nn = GRCL(64, 64, 5, 3, 1, 1)
        self.pool2_nn = nn.MaxPool2d(2, 2)
        self.grcl2_nn = GRCL(64, 128, 5, 3, 1, 1)
        self.pool3_nn = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.grcl3_nn = GRCL(128, 256, 5, 3, 1, 1)
        self.pool4_nn = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv_final_nn = nn.Conv2d(256, 512, 2, 1, 0)
        self.bn_final_nn = nn.BatchNorm2d(512)
        self.out_nn = nn.ReLU(True)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),  # 512
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        conv1 = self.conv1_nn(input)
        act1 = self.act1_nn(conv1)
        pool1 = self.pool1_nn(act1)
        grcl1 = self.grcl1_nn(pool1)
        pool2 = self.pool3_nn(grcl1)
        grcl2 = self.grcl2_nn(pool2)
        pool3 = self.pool3_nn(grcl2)
        grcl3 = self.grcl3_nn(pool3)
        pool4 = self.pool4_nn(grcl3)
        conv_final = self.conv_final_nn(pool4)
        bn_final = self.bn_final_nn(conv_final)
        out = self.out_nn(bn_final)  # the out of conv

        b, c, h, w = out.size()
        assert h == 1, "the height of conv must be 1"
        out = out.squeeze(2)
        out = out.permute(2, 0, 1)

        # rnn features
        output = self.rnn(out)

        return output
