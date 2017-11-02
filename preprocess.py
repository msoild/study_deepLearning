# Created by wangliang on 17-11-02.
# encoding=utf-8
import sys
import os
from scipy import misc
import numpy as np




def main():
    l = len(sys.argv)
    if l < 3:  # 检查参数的数量是否足够
        print'eg: python img2pkl.py list.txt dst.npy\n' \
             'convert image to npy\n'
        return

    data_dir = os.path.dirname(os.getcwd())+'/'+sys.argv[1]
    src = sys.argv[2]
    dst = sys.argv[3] if l > 3 else 'data.pkl'
    print data_dir+src,"src"
    with open(data_dir+src, 'r') as f:  # 读取图片列表
        list = f.readlines()

    data = []
    labels = []
    for i in list:
        name, label = i.strip('\n').split(' ')  # 将图片列表中的每一行拆分成图片名和图片标签
        print name + ' processed'
        name = data_dir+name
        img = misc.imread(name)  # 将图片读取出来，存入一个矩阵
        img /= 255  # 将图片转换为只有0、1值的矩阵
        img.resize((img.size, 1))  # 为了之后的运算方便，我们将图片存储到一个img.size*1的列向量里面
        data.append(img)
        labels.append(int(label))

    print 'write to npy'
    np.save(data_dir+dst, [data, labels])  # 将训练数据以npy的形式保存到成本地文件
    print 'completed'


if __name__ == '__main__':
    main()