# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, AI
# All rights reserved.
# @Time 2019/11/1

import sys
import numpy as np
import visdom
from backports import subprocess

python_path = sys.executable
VisdomExceptionBase = ConnectionError


class Visualizer():
    def __init__(self):

        # 子进程去启动visdom
        server = subprocess.Popen(
            [python_path, "-m", "visdom.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        print("Starting visdom server...")
        print("If the program is stuck here, please check your visdom installation."
              " Run 'python -m visdom.server' in your shell ")
        while True:
            line = server.stdout.readline()
            if 'http' in line:
                break
        print("Starting visdom server succeeded!!!")
        print("Please go to http://localhost:8097/")

        # 使用默认的local host, port 8097
        self.vis = visdom.Visdom()
        self.display_id = 1  # 创建窗口的时候必须指定，否则每次会自动+1

    def plot_current_losses(self, epoch, counter_ratio, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        # Bug report
        # 当visdom检测到Y 是 n*1 形式的， 一定会默认做reshape
        # 见 if Y.ndim == 2 and Y.shape[1] == 1:
        #       Y = Y.reshape(1, Y.shape[0])
        #       X = X.reshape(X.shape[0])
        #
        X = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
        Y = np.array(self.plot_data['Y'])
        if X.shape[1] == 1 and Y.shape[1] == 1:
            X = X.squeeze(1)
            Y = Y.squeeze(1)

        try:
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': 'loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win = self.display_id
            )
        except VisdomExceptionBase:
            print('\n\nCould not connect to Visdom server')
            exit(1)

