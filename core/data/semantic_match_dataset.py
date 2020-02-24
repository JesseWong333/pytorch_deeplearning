# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/11/14
#

# 装载pairs.csv 数据对, 我不创造Dynamic batch 数据

import torch
import numpy as np
import json
import torch.utils.data as data


class SemanticMatchDataset(data.Dataset):
    def __init__(self, csv_pair_file, item_file):
        with open(csv_pair_file, mode='r', encoding='utf-8') as f:
            self.pairs = [line.strip() for line in f.readlines()]

        self.ori_data = self.read_json(os.path.join(BASE_PATH, 'senior_math_processed.json'))
        pass

    def read_json(self, js_f):
        with open(js_f, 'r', encoding='utf8') as f:
            lines = f.readlines()
            if len(lines) > 0:
                phy_dict = dict()
                for l in lines:
                    single = json.loads(l)
                    phy_dict.update(single)
            else:
                phy_dict = json.loads(lines)
        return phy_dict

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        pass

    @staticmethod
    def collate_fn(batch):
        pass


if __name__ == '__main__':
    import os
    BASE_PATH = '/media/Data/wangjunjie/a_NLP/similarity_question'
    csv_pair_file = os.path.join(BASE_PATH, 'pairs.csv')
    item_file = os.path.join(BASE_PATH, 'senior_math_processed.json')
    SemanticMatchDataset(csv_pair_file, item_file)
