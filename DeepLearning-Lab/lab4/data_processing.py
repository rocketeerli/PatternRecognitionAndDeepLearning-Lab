import torch
import numpy as np
import codecs
import numpy.random

# 读取字典
def readDict(fileName):
    dic=[]
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            b=line.split(' ')         # 将每一行以空格为分隔符转换成列表
            key = b[0]
            val = [float(i) for i in b[1:]]
            dic.append([key, val])
        dic=dict(dic)
    return dic

# 读取训练集和测试集，进行数据预处理
def readData(posFile, negFile, dic):
    # 读取 pos 文件
    pos_set = []
    with open(posFile, 'r', encoding="windows-1252", errors='ignore') as f:
        for line in f.readlines():
            words = line.split(' ')         # 将每一行以空格为分隔符转换成列表
            # 每个单词
            sentence = dealWords(words, dic)
            pos_set.append(sentence)
    # 读取 neg 文件
    neg_set = []
    with open(negFile, 'r', encoding="windows-1252", errors='ignore') as f:
        for line in f.readlines():
            words = line.split(' ')         # 将每一行以空格为分隔符转换成列表
            # 每个单词
            sentence = dealWords(words, dic)
            neg_set.append(sentence)
    train_set = pos_set[:4000] + neg_set[:4000]
    target = [1 for i in range(4000)] + [0 for i in range(4000)]
    test_set = pos_set[4000:] + neg_set[4000:]
    test_target = [1 for i in range(1331)] + [0 for i in range(1331)] 
    # 转成 numpy 格式返回
    return np.array(train_set), np.array(target), np.array(test_set), np.array(test_target)

# 处理每一行数据
def dealWords(words, dic):
    sentence = []
    ite = 0
    # 前面补 0 
    zero_list = [0 for i in range(50)]
    for word in words:
        if ite >= 50 :
            break
        if word in dic.keys():
            sentence.append(dic[word])
            # print(word)
            # print(len(list(dic[word])))
            ite += 1
    for i in range(50-ite):
        sentence.insert(0, zero_list)
    return sentence

# 打乱顺序
def change_order(set, target):
    # 打乱顺序
    permutation = np.random.permutation(target.shape[0])
    # index = [i for i in range(len(set))]  
    # np.random.shuffle(index) 
    return set[permutation, :, :], target[permutation]

def get_data():
    dic = readDict('./glove.6B.50d.txt')  # 读取单词字典
    train_set, target, test_set, test_target = readData('rt-polarity.pos', 'rt-polarity.neg', dic)
    train_set, target, test_set, test_target = torch.Tensor(train_set), torch.Tensor(target), torch.Tensor(test_set), torch.Tensor(test_target)
    # 更改顺序
    train_set, target = change_order(train_set, target)
    test_set, test_target = change_order(test_set, test_target)
    return train_set, target, test_set, test_target

if __name__ == '__main__':
    get_data()

