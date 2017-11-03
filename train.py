# encoding=utf-8
from layers import *
from loss import *
from fc_network import *
from accuracy import *
from  abs_pwd import *



def main():
    base_pwd = get_abs_pwd('data/alphabet/')
    train_datalayer = Data(base_pwd+'train.npy', 1024)  # 用于训练，batch_size设置为1024
    test_datalayer = Data(base_pwd+'validate.npy', 10000)  # 用于验证，所以设置batch_size为10000,一次性计算所有的样例

    losslayer = QuadraticLoss()
    accuracy = Accuracy()
    rate=1000.0
    net = FcNetwork([289, 26], losslayer)
    epochs = 100

    # train(self, train_datalayer, test_datalayer, accuracy,rate, epochs)
    net.train(train_datalayer, test_datalayer, accuracy, rate, epochs)


if __name__ == '__main__':
    main()