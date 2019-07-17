import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from dataset import Cifar10Dataset
import torchvision

class VGG(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 第一层
            # 取卷积核为3*3，补边为1，输入通道为3，输出通道为64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # 批标准化
            # nn.BatchNorm2d(64, affine=True),
            # 激活函数
            nn.ReLU(inplace=True),
            # 池化，核为2*2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            # 取卷积核为3*3，补边为1，输入通道为64，输出通道为128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            # 取卷积核为3*3，补边为1，输入通道为128，输出通道为256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五层
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 一层全连接层，输入512层，输出10（10类）
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(epoch, model, writer, train_loader, optimizer, loss_func, device):
    print("train:")
    # loss损失
    train_loss = 0
    # 准确个数
    train_correct = 0
    total = 0
    i = 0
    for batch_num, (data, target) in enumerate(train_loader):
        # 训练数据和其对应的标签
        data, target = data.to(device), target.to(device)
        # 清空上次训练的梯度
        optimizer.zero_grad()
        # 将训练数据填入模型中，进行预测类别，前向传播
        output = model(data)
        # 计算损失函数
        loss = loss_func(output, target)
        # 反向传播
        loss.backward()
        # 根据计算的梯度更新网络的参数
        optimizer.step()
        train_loss += loss.item()
        # 生成每个数据预测的类别，即概率最大的
        prediction = torch.max(output, 1)
        total += target.size(0)
        # 计算准确个数，以用来计算准确率
        train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        # 每100个batch计算一次平均的loss，绘制loss曲线，曲线名为train_loss
        if i % 100 == 99:
            avg_loss = train_loss / 100
            writer.add_scalar('train_loss', avg_loss, epoch * len(train_loader) + i + 1)
            train_loss = 0.0
        i += 1
    # 返回每次训练，针对于训练集的预测准确率
    return 100. * train_correct / total


def exam_model(epoch, model, writer, test_loader, loss_func, device):
    print("test:")
    # loss损失
    test_loss = 0
    # 预测正确的个数
    test_correct = 0
    total = 0
    i = 0
    for batch_num, (data, target) in enumerate(test_loader):
        # 测试数据和其对应的标签
        data, target = data.to(device), target.to(device)
        # 将测试数据填入模型中，进行预测类别
        output = model(data)
        # 计算损失函数
        loss = loss_func(output, target)
        test_loss += loss.item()
        # 生成每个数据预测的类别，即概率最大的
        prediction = torch.max(output, 1)
        total += target.size(0)
        # 计算准确个数，以用来计算准确率
        test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        # 每25个batch计算一次平均的loss，绘制loss曲线，曲线名为test_loss
        if i % 25 == 24:
            avg_loss = test_loss / 25
            # writer.add_scalar('test_loss', avg_loss, epoch * len(test_loader) + i + 1)
            test_loss = 0.0
        i += 1
    # 每次测试将计算的准确率绘制为曲线test_accuracy
    writer.add_scalar('test_accuracy', 100. * test_correct / total, epoch + 1)
    # 返回每次测试，针对于测试集的预测准确率
    return 100. * test_correct / total


def choice(model):
    print('=====================================')
    print('===             请选择优化方式    ===')
    print('===             1.Adam            ===')
    print('===             2.SGD             ===')
    print('===             3.RMSprop         ===')
    print('=====================================')
    choices = int(input())
    if choices == 1:
        # 选用Adam作为优化器，学习率为0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif choices == 2:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    elif choices == 3:
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    return optimizer


# 运行函数，进行模型训练和测试集的测试
def run(device):
    # 利用tensorboardX中的SummaryWriter类可视化测试集准确率曲线
    writer = SummaryWriter('./logs_vgg')
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    # # 利用自定义 Dataset
    # train_set = Cifar10Dataset(root='./Cifar-10', train=True, transform=transform)  # 训练数据集
    # test_set = Cifar10Dataset(root='./Cifar-10', train=False, transform=transform)
    # 利用库函数进行数据集加载
    train_set = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=train_transform)  # 训练数据集
    test_set = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=test_transform)
    # 将数据集整理成batch的形式并转换为可迭代对象，200行
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=200, shuffle=False)
    # 调用编写好的vgg11模型，对环境进行配置
    cudnn.benchmark = True
    model = VGG().to(device)
    # 选择合适的优化器
    optimizer = choice(model)
    # 计算损失函数
    loss_func = nn.CrossEntropyLoss().to(device)
    # 设置训练次数为5次
    epochs = 5
    accuracy = 0
    # 进行5次训练，每进行一次训练，就用测试集进行一次测试
    for epoch in range(1, epochs + 1):
        print('\n===> epoch: %d/5' % epoch)
        train_result = train(epoch - 1, model, writer, train_loader, optimizer, loss_func, device)
        print('*****训练集预测准确率为: %.3f%%' % train_result)
        test_result = exam_model(epoch - 1, model, writer, test_loader, loss_func, device)
        print('*****测试集预测准确率为: %.3f%%' % test_result)
        accuracy = max(accuracy, test_result)
        # 保存训练好的模型
        if epoch == epochs:
            print('*****测试集最高准确率为: %.3f%%' % accuracy)
            torch.save(model, 'vgg11.pt')
