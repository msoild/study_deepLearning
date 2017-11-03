# encoding=utf-8
import numpy as np


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, x, label):
        self.x = x
        self.label = np.zeros_like(x)  # 由于我们的label本身只包含一个数字，我们需要将其转换成和模型输出值尺寸相匹配的向量形式
        for a, b in zip(self.label, label):
            a[b] = 1.0  # 只有正确标签所代表的位置概率为1，其他为0
            loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2  # 求平均后再除以2是为了表示方便
            self.y=loss
        return self.y

    def backward(self):
        self.dx = (self.x - self.label) / self.x.shape[0]  # 2被抵消掉了
        return self.dx

