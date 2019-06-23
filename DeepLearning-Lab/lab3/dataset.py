import torch.utils.data
import pickle
import numpy as np
from PIL import Image


# 需要继承data.Dataset
class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        """
        1. 初始化数据路径
        2. 读取数据
        3. 整理数据
        """
        self.root = root + "\\cifar-10-batches-py\\"     # 目录
        self.fileName = ""
        self.data = []       # 存储图片数据
        self.targets = []    # 存储图片标签
        # 存储 transform 
        self.transform = transform
        self.target_transform = target_transform

        # 读取数据
        if train :  # 训练集
            self.fileName = "data_batch_"
            for i in range(1, 6) :
                file = self.root + self.fileName + str(i)  # 构建文件路径
                self.save(file)
        else :      # 测试集
            self.fileName = "test_batch"          # test 文件路径
            file = self.root + self.fileName
            self.save(file)
        # 整理读取的数据
        self.data = np.vstack(self.data)   # 合并二维数组，并转成 numpy.ndarray 类型，方便 reshape
        self.data = self.data.reshape(-1, 3, 32, 32)   
        # print(self.data.shape)    # (50000, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # 转置 转成 HWC 格式
        # print(self.data.shape)    # (50000, 32, 32, 3)

    def __getitem__(self, index):
        """
        1. 按照下标 (index) 读数据
        2. 转换图片格式.
        3. 返回图片数据和标签信息
        """
        img, target = self.data[index], self.targets[index]
        # 将图片信息转成 PIL.Image.Image 类型
        img = Image.fromarray(img)
        # 对 PIL.Image 进行变换
        if self.transform is not None:
            img = self.transform(img)
        # 对标签格式进行转换
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        # 返回数据长度
        return len(self.data)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def save(self, file):
        dict = self.unpickle(file)            # 利用官网提供的方法读取数据，存储在字典中
        self.data.append(dict[b'data'])       # 将数据集中的 data 存储起来
        self.targets.extend(dict[b'labels'])  # list 类型
