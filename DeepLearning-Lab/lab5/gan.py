from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import draw

# 常量定义
BATCH_SIZE = 150       # batch 的大小
EPOCH_NUM = 2000         # epoch 的数量
TRAIN_SET_SIZE = 6000  # 训练集的大小
INPUT_SIZE = 6         # 生成器输入的噪声维度
LR = 0.0003            # 两个网络的学习率

# 加载数据集
m = loadmat("./points.mat")
data = m['xx']
# 打乱顺序
np.random.shuffle(data)
# 拆分成训练集和测试集 
train_set = data[:TRAIN_SET_SIZE]
test_set = data[TRAIN_SET_SIZE:]

# 使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取坐标点
def get_min_data():
    x_min = np.min(test_set[:, 0])
    x_max = np.max(test_set[:, 0])
    y_min = np.min(test_set[:, 1])
    y_max = np.max(test_set[:, 1])
    return x_min, x_max, y_min, y_max

# 生成网络
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 2)
            # nn.Tanh()  # 加上该层会导致分布被截断
        )
 
    def forward(self, x):
        x = self.gen(x)
        return x


# 判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # 两层线性全连接
        self.dis = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.dis(x)
        return x

# 定义网络
D = discriminator().to(device)
G = generator().to(device)
# Binary cross entropy loss and optimizer
d_optimizer = torch.optim.SGD(D.parameters(), lr=LR)
g_optimizer = torch.optim.SGD(G.parameters(), lr=LR)

# 训练
def train():
    for epoch in range(EPOCH_NUM):
        for i in range(int(TRAIN_SET_SIZE / BATCH_SIZE)):
            label = torch.from_numpy(train_set[i*BATCH_SIZE: (i+1)*BATCH_SIZE]).float().to(device)
            G_input = torch.randn(BATCH_SIZE, INPUT_SIZE).to(device)
            G_out = G(G_input).to(device)
            # 计算判别器判别的概率
            prob_label = D(label)
            prob_gen = D(G_out)
            # 计算生成器和判别器的 Loss
            D_loss = - torch.mean(torch.log(prob_label) + torch.log(1. - prob_gen)).to(device)
            G_loss = torch.mean(torch.log(1. - prob_gen)).to(device)
            # 更新判别器
            d_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            d_optimizer.step()
            # 更新生成器
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()
            print("epoch: %d \t batch: %d \t\t d_loss: %.8f \t g_loss: %.8f "%(epoch+1, i+1, D_loss, G_loss))
        if (epoch+1) % 5 == 0 :
            cb = test_G()
            plt.savefig('./result/gan/epoch'+ str(epoch+1))
            cb.remove()
            plt.cla()

# 测试训练好的生成器
def test_G():
    G_input = torch.randn(1000, INPUT_SIZE).to(device)
    G_out = G(G_input)
    G_data = np.array(G_out.cpu().data)
    # 计算坐标点边界
    x_min, x_max, y_min, y_max = get_min_data()
    x_min = np.min(np.append(G_data[:, 0], x_min))
    x_max = np.max(np.append(G_data[:, 0], x_max))
    y_min = np.min(np.append(G_data[:, 1], y_min))
    y_max = np.max(np.append(G_data[:, 1], y_max))
    # 画背景
    cb = draw.draw_background(D, x_min, x_max, y_min, y_max)
    # 画出测试集的点分布和生成器输出的点分布
    draw.draw_scatter(test_set, 'b', x_min, x_max, y_min, y_max)
    draw.draw_scatter(G_data, 'r', x_min, x_max, y_min, y_max)
    return cb

if __name__ == '__main__':
    train()
    test_G()
    
