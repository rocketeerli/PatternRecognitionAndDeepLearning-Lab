'''
python 版本: 3.7.3
pytorch 版本: 1.1
实验平台: Windows10
'''
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 定义神经网络
class SineWithLSTM(nn.Module):
    def __init__(self):
        super(SineWithLSTM, self).__init__()
        # 第一层 lstm，input: 1 维，hidden: 51 维
        self.lstm1 = nn.LSTMCell(1, 51)
        # 第二层 lstm, input:51维, hidden: 51 维
        self.lstm2 = nn.LSTMCell(51, 51)
        # 全连接层
        self.linear = nn.Linear(51, 1)
 
    def forward(self, input, future = 0):
        outputs = []
        # 初始化两个 LSTM
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        # 分成多个 batch
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t = input_t.to(device)
            h_t, c_t = h_t.to(device), c_t.to(device)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = h_t2.to(device), c_t2.to(device)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        # 进行预测，默认 future 为 0，不进行预测
        for i in range(future):
            output = output.to(device)
            h_t, c_t = h_t.to(device), c_t.to(device)
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = h_t2.to(device), c_t2.to(device)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

# 画线
def draw(y, i):
    plt.figure(figsize=(30,10))
    plt.title('sine predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    plt.plot(np.arange(999), y[1][:999], linewidth = 2.0)
    plt.plot(np.arange(999, np.shape(y[1])[0]), y[1][999:], linestyle=':', linewidth = 2.0)
    plt.savefig('predict%d.jpg'%i)
    plt.close()

# 反向传播，利用 LBFGS 优化器进行优化
def closure():
    optimizer.zero_grad()
    out = seq(input)
    loss = criterion(out, target).to(device)
    print('loss:', loss.item())
    loss.backward()
    return loss

# 训练
def train():
    for i in range(5):
        print('STEP: ', i)
        optimizer.step(closure)
        # 开始预测
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().cpu().numpy()
        # 画线
        draw(y, i)

if __name__ == '__main__':
    # 加载训练集和测试集
    data = torch.load('./traindata.pt')
    # 将数据转成 tensor 格式，并放到 gpu 上运行
    input = torch.from_numpy(data[3:, :-1]).to(device)
    target = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)
    # 创建神经网络，也放到 gpu 上
    seq = SineWithLSTM().to(device)
    seq.double()
    criterion = nn.MSELoss()
    # 使用 LBFGS 优化器
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # 训练
    train()


