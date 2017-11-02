# encoding=utf-8

import numpy as np

class FullyConnect:
    def __init__(self, l_x, l_y):  # 两个参数分别为输入层的长度和输出层的长度
        self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)
        self.bias = np.random.randn(l_y, 1)  # 使用随机数初始化参数
        self.lr = 0  # 先将学习速率初始化为0，最后统一设置学习速率

    def forward(self, x):
        self.x = x  # 把中间结果保存下来，以备反向传播时使用
        self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])  # 计算全连接层的输出
        return self.y  # 将这一层计算的结果向前传递

    def backward(self, d):
        ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]  # 根据链式法则，将反向传递回来的导数值乘以x，得到对参数的梯度
        self.dw = np.sum(ddw, axis=0) / self.x.shape[0]
        self.db = np.sum(d, axis=0) / self.x.shape[0]
        self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])

        # 更新参数
        self.weights -= self.lr * self.dw
        self.bias -= self.lr * self.db
        return self.dx  # 反向传播梯度

    def dump(self):
        print 'W:\t%s\nb:\t%s'%(self.W, self.b)