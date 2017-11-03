from loss import *
import numpy as np

def getDataset():
    x = np.array([[0, 1, 3], [2, 3, 1], [3, 5, 6]])
    y = np.array([0, 2, 1])
    return x, y


class QuadraticLoss:
    def __init__(self):
        pass

    def forward(self, y, label):
        self.y = y
        self.label = np.zeros_like(y)  # ?????label???????????????????????????????????
        for a, b in zip(self.label, label):
            a[b] = 1.0  # ???????????????1????0
            print a, "a"
        # self.loss = np.sum(np.square(y - self.label)) / self.y.shape[0] / 2  # ???????2???????
        # return self.loss

    def backward(self):
        self.dy = (self.y - self.label) / self.y.shape[0]  # 2?????
        return self.dy



if __name__ == '__main__':
    x, y = getDataset()
    quad = QuadraticLoss()
    quad.forward(x, y)
