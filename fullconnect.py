# encoding=utf-8

import numpy as np


class FullyConnect:
    def __init__(self, l_x, l_y, activator):  # 两个参数分别为输入层的长度和输出层的长度

        self.W = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.b = np.random.randn(l_y, 1)  # 使用随机数初始化参数
        self.activator = activator

    def forward(self, x):
        '''
        :param x:
        :return: 1*m
        '''
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        y = []
        for xx in x:
            y.append(self.activator.forward(
                np.dot(self.W, xx) + self.b))
        self.y = np.array(y)
        return self.y # 将这一层计算的结果向前传递

    def backward(self, d):
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(self.activator.backward(d), self.x)]  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.W.T, dd) for dd in d])

        # 更新参数

        return self.dx  # 反向传播梯度

    def update(self, lr):
        self.W -= lr * self.dw
        self.b -= lr * self.db

    def dump(self):
        print 'W:\t%s\nb:\t%s'%(self.W, self.b)