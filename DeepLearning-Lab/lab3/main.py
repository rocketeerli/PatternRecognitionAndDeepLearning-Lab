"""
系统环境：Windows10
python版本：3.6
pytorch版本：torch1.1.0
cuda版本：9.1
cudnn版本：7.1
tensorflow版本：1.8.0
程序运行：在命令行中输入python main.py，利用argparse设置一个cuda参数，默认值'GPU',还可以通过python main.py --cuda 'CPU'选择CPU运行，
         然后先选择是否执行vgg11网络，执行vgg11时可以选择优化方法；然后继续选择是否执行resNet
         其中vgg中的BN层被注释掉
可视化图像：将测试集准确率和训练集训练时的loss图像进行可视化，分别为test_accuracy和train_loss曲线，vgg的存在logs_vggw文件夹下，resnet的存在logs_resNet文件夹下
"""

import torch.utils.data
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='GPU', type=str, choices=['GPU', 'CPU'], help='选择使用cpu/gpu')
    args = parser.parse_args()
    # CPU或GPU运行
    if args.cuda == 'GPU':
        device = torch.device('cuda')
    elif args.cuda == 'CPU':
        device = torch.device('cpu')
    # 选择是否执行vgg11
    choicess = input('请选择是(1),否(0)运行vgg-11\nYour choice:')
    if choicess == '1':
        import vgg
        vgg.run(device)
    # 选择是否执行resNet18
    choicess_1 = input('请选择是(1),否(0)运行ResNet-18\nYour choice:')
    if choicess_1 == '1':
        import resNet
        resNet.run(device)
