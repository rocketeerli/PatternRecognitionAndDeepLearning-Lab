# 实验内容

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
	
# 代码运行

1. 执行命令：

	> python .\main.py

	默认使用 gpu 运行，如果想用 CPU 运行，执行命令 `python .\main.py --cuda CPU` 

2. 首先选择是否执行 VGG-11； 然后再选择是否执行 ResNet

3. 如果执行 VGG-11， 需要选择优化器（三种优化器选择：Adam、 SGD、 RMSprop）

# 实验结果

## 执行命令查看实验的loss曲线和准确率曲线

1. 查看 VGG-11 实验结果

> tensorboard --logdir=./logs_vgg

准确率曲线

![VGG测试准确率曲线](https://img-blog.csdnimg.cn/20190717225218683.png)

Loss 曲线

![VGG测试Loss曲线](https://img-blog.csdnimg.cn/2019071722553918.png)

2. 查看 ResNet 实验结果

> tensorboard --logdir=./logs_resNet

准确率曲线

![ResNet测试准确率曲线](https://img-blog.csdnimg.cn/20190717225827668.png)

Loss 曲线

![ResNet测试Loss曲线](https://img-blog.csdnimg.cn/20190717230052289.png)

# 备注

1. 代码运行的比较慢，还是用 gpu 速度快一些
2. 实验是和队友组队完成的，VGG-11 是队友负责的；ResNet 网络是我负责的（已经询问过队友公开代码的问题了，版权意识还是要有的，哈哈）
3. 实验很多内容都是参考网上的前辈们的，在此感谢各位前辈们的分享。本打算也来写点博客做下贡献，奈何学艺不精，怕误人子弟