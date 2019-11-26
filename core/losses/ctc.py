# encoding: utf-8

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/8/20

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_loss

@register_loss
class CTCLoss(nn.Module):
    """

    """
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=True)

    def forward(self, text, length, preds):
        preds = preds.log_softmax(2)
        batch_size = preds.shape[0]
        # print(batch_size)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(preds.device)
        preds = preds.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        ctc_loss = self.criterion(preds, text, preds_size, length)
        torch.backends.cudnn.enabled = True
        return ctc_loss


if __name__ == '__main__':
    ctc_loss = CTCLoss()
    preds = torch.randn(10, 40, 11)
    text = torch.randint(low=1, high=10, size=160, dtype=torch.int32)
    target_lengths = torch.randint(low=1, high=16, size=(10,), dtype=torch.int32)
    ctc_loss(text, target_lengths, preds)