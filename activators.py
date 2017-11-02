# encoding=utf-8

import numpy as np

class Sigmoid:
    def __init__(self):  # 无参数，不需初始化
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.y = self.sigmoid(x)
        return self.y

    def backward(self, d):
        sig = self.sigmoid(self.x)
        self.dx = d * sig * (1 - sig)
        return self.dx  # 反向传递梯度