# encoding=utf-8

from fullconnect import *
from activators  import *


class FcNetwork():
    def __init__(self,layers_nodes_num, top_layer):
        '''

        :param layers_nodes_num:输入及隐含层
        :param top_layer: 输出层，主要是精确层也可能是其他函数
        '''
        self.layers=[]
        for i in range(len(layers_nodes_num)-1):
            self.layers.append(FullyConnect(layers_nodes_num[i],
                                            layers_nodes_num[i+1], Sigmoid()))
        self.top_layer = top_layer

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.y

        return output

    def calc_gradient(self):
        d = self.top_layer.backward()  # 调用损失层backward函数层计算将要反向传播的梯度
        for layer in self.layers[::-1]:  # 反向传播
            d = layer.backward(d)

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def train(self, train_datalayer, test_datalayer, accuracy,rate, epochs):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epochs):
            print 'epochs:', i
            losssum = 0
            iters = 0
            while True:
                data, pos = train_datalayer.forward()  # 从数据层取出数据
                x, label = data
                output = self.predict(x)  # 调用损失层forward函数计算损失函数值
                loss = self.top_layer.forward(output,label)
                self.calc_gradient()
                self.update_weight(rate)
                losssum += loss
                iters += 1
                if pos == 0:  # 一个epoch完成后进行准确率测试
                    data, _ = test_datalayer.forward()
                    x, label = data
                    test_y=self.predict(x)
                    accu = accuracy.forward(test_y, label)  # 调用准确率层forward()函数求出准确率
                    print 'loss:', losssum / iters
                    print 'accuracy:', accu
                    break




