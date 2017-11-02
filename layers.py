# encoding=utf-8
import numpy as np

class Data:
    def __init__(self, name, batch_size):  # 数据所在的文件名name和batch中图片的数量batch_size
        with open(name, 'rb') as f:
            data = np.load(f)
        self.x = data[0]  # 输入x
        self.y = data[1]  # 预期正确输出y
        self.l = len(self.x)
        self.batch_size = batch_size
        self.pos = 0  # pos用来记录数据读取的位置

    def forward(self):
        pos = self.pos
        bat = self.batch_size
        l = self.l
        if pos + bat >= l:  # 已经是最后一个batch时，返回剩余的数据，并设置pos为开始位置0
            ret = (self.x[pos:l], self.y[pos:l])
            self.pos = 0
            index = range(l)
            np.random.shuffle(index)  # 将训练数据打乱
            self.x = self.x[index]
            self.y = self.y[index]
        else:  # 不是最后一个batch, pos直接加上batch_size
            ret = (self.x[pos:pos + bat], self.y[pos:pos + bat])
            self.pos += self.batch_size

        return ret, self.pos  # 返回的pos为0时代表一个epoch已经结束

    def backward(self, d):  # 数据层无backward操作
        pass
