# 介绍

本仓库为哈尔滨工业大学 2019 春模式识别与深度学习的实验+课件

* Pattern-Recognition-Lab 部分是模式识别的实验，一个实验，只放了代码+报告，没有放数据集

* DeepLearning-Lab 部分是深度学习的实验，一共 6 个实验，最后一次实验自由发挥，没有放到仓库里

# Pattern-Recognition-Lab（模式识别实验）

### 实验内容

1. 从 SVM, PCA, GMM, HMM 四个练习中选三个进行训练，可以调用各种库函数，要求实现完整的功能

2. 或是实现一个复杂的系统，要求使用机器学习的知识

### 说明

我做的是第一个，选择了 SVM, PCA, GMM 三个项目。

其中 GMM 项目参考：[自制数据集实现动物种类语音识别](https://github.com/LiGuiye/MFCC-GMM-python2.7)

# DeepLearning-Lab（深度学习实验）

也是第一次写深度学习的代码，一路磕磕绊绊，还好全部完成了。

代码还有很多改进的空间，留待以后慢慢探索。

## 实验一（MLP）

* 使用PyTorch实现MLP，并在MNIST数据集上验证

没什么好说的，深度学习的入门项目，体验一下就好。

据助教说，这次实验不算成绩，也不知道最后算没算。不过这都无所谓了，重在学习，分数次要（哈哈，千万别当真）

## 实验二（AlexNet）

* 基于 PyTorch 实现 AlexNet
* 在 Cifar-10 数据集上进行验证
* 使用tensorboard进行训练数据可视化（Loss 曲线）
* 如有条件，尝试不同参数的影响，尝试其他网络结构
* 请勿使用torchvision.models.AlexNet

**注： 更改后的 AlexNet，但结构相似。**主要目的还是为了体验一下卷积神经网络，因此为了不使用太大的数据集，要更改一下参数

## 实验三（VGG/ResNet）

* 基于PyTorch实现 VGG/ResNet 结构，并在 Cifar-10 数据集上进行验证
	1. VGG要求实现VGG-11（Conv部分按论文实现，Classifier直接一层全连接即可）
	2. ResNet要求实现ResNet-18。均要求5个epoch以内达到50%的测试集精度。

* 基于 VGG 进行训练方式对比（LR 和优化器的对比）
	1. LR对比三组及以上【此时选用任一种优化器】
	2. 优化器对比 SGD 与 Adam【选用LR对比时的最佳LR】

* 基于 CUDA 实现，并设置 argparse 参数
	可设定是否使用GPU，通过 argparse 包实现，默认参数设定为GPU

* 自定义实现 dataset
	基于opencv或Pillow
	
## 实验四（RNN）

* 对Sine函数进行预测
	给定部分序列的值，进行后续序列值的预测

* 基于影评数据进行文本情感预测
	基于影评文本，进行文本情感分析

实验分为两部分，分别使用了 LSTM 和 GRU：

1. 第一部分实现 sine 函数预测，这是 Pytorch 官网的一个例子：[sine 例子](https://github.com/pytorch/examples/blob/master/time_sequence_prediction/generate_sine_wave.py)

参考博客：[Pytorch LSTM 时间序列预测](https://blog.csdn.net/duan_zhihua/article/details/84491672)

2. 第二部分实现文本情感分析，网络结构参考：[使用 PyTorch RNN 进行文本分类](https://www.jianshu.com/p/46d9dec06199)

这部分做的时间很匆忙，我觉得可改进的空间很大。

## 实验五（GAN）

* 基于PyTorch实现生成对抗网络
	1. 拟合给定分布
	2. 要求可视化训练过程
	
* 对比GAN、WGAN、WGAN-GP（稳定性、性能）

* 对比不同优化器的影响

这次实验算是我做的最顺利的一次实验了，没有出什么 bug，实验效果很好，很神奇。GAN 初体验，感觉很强大

## 实验六（自选项目，组队完成，暂时没有放到仓库里）

### 基于PyTorch完成一个项目

选择范围：

* 去噪(Denoising)
* 超分辨率(Super-resolution)
* 检测(Detection)
* 分割(Segmentation)

本来想打算从后两个高层视觉中选，后来发现太难了，，，去噪的题目上学期图像处理已经做过了（虽然不是神经网），于是选择了超分辨率。

使用的是 SRGAN，参考:[PyTorch-SRGAN](https://github.com/aitorzip/PyTorch-SRGAN)

# 最后

初次体验神经网络，很神奇。

代码肯定有很多不好的地方，以后慢慢改

建仓库的目的，主要还是为了以后写代码，我觉得课程的实验真的让我学到了很多，在这里保存代码，方便以后借鉴
