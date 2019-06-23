'''
python 版本: 3.7.3
pytorch 版本: 1.1
实验平台: Windows10
'''
import torch
import numpy as np
import data_processing

# 使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义神经网络
class AnalyWithGRU(torch.nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
        super(AnalyWithGRU, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        # 指定 GRU 的各个参数
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        # 最后一层：一个线性层，全连接，用于分类
        self.out = torch.nn.Linear(hidden_size, out_size)
        
    def forward(self, word_inputs, hidden):
        '''
        batch_size:  batch 的大小，这里默认是1，表示一句话
        word_inputs: 输入的向量
        hidden: 上下文输出
        '''
        # resize 输入的数据
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)
        output, hidden = self.gru(inputs, hidden)
        output = self.out(output)
        # 仅返回最后一个向量,用 RNN 表示
        output = output[:,-1,:]
        return output, hidden
    def init_hidden(self):
        # 每次第一个向量没有上下文，在这里返回一个上下文
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden

# 训练
def train():
    '''
    隐层 50: 表示词向量的宽度
    输出 2 : 隐层用，输出是分类的个数
    '''
    # 获取训练集和测试集
    train_set, target, test_set, test_target = data_processing.get_data()
    net = AnalyWithGRU(50, 2).to(device)    # 定义神经网络
    criterion = torch.nn.CrossEntropyLoss()         # 设置 loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 设置优化器
    for epoch in range(3):
        for i in range(train_set.size()[0]):
            encoder_hidden = net.init_hidden()
            input_data = torch.autograd.Variable(train_set[i])
            output_labels = torch.autograd.Variable(torch.LongTensor([target[i]]))
            input_data, encoder_hidden = input_data.to(device), encoder_hidden.to(device)
            # 训练
            encoder_outputs, encoder_hidden = net(input_data, encoder_hidden)
            # 优化器
            optimizer.zero_grad()
            loss = criterion(encoder_outputs, output_labels.to(device)).to(device)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0 or i==train_set.size()[0]-1:
                print("epoch: " + str(epoch+1) + "\t" + "loss: " + str(loss.item()))
                Accuracy(net, test_set, test_target)
    return

# 在测试集上进行测试，并计算准确率
def Accuracy(net, test_set, test_target):
    # 使用测试数据测试网络
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(test_set.size()[0]):
            encoder_hidden = net.init_hidden()
            input_data = torch.autograd.Variable(test_set[i])
            labels = torch.autograd.Variable(torch.LongTensor([test_target[i]]))
            input_data, labels = input_data.to(device), labels.to(device) # 将输入和目标在每一步都送入GPU
            outputs, _ = net(input_data, encoder_hidden.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 1000 test dataset: %d %%' % (
        100 * correct / total))
    return 100.0 * correct / total

if __name__ == '__main__':
    train()
