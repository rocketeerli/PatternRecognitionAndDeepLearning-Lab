# 实验内容

使用PyTorch实现MLP，并在MNIST数据集上验证

# 介绍

这是深度学习课程的第一个实验，主要目的就是熟悉 Pytorch 框架。MLP 是多层感知器，我这次实现的是四层感知器，个人认为，感知器的代码大同小异，尤其是用 Pytorch 实现，除了层数和参数外，代码都很相似。

Pytorch 写神经网络的主要步骤主要有以下几步：

1. **构建网络结构**
2. **加载数据集**
3. **训练神经网络（包括优化器的选择和 Loss 的计算）**
4. **测试神经网络**

# 过程
### 构建网络结构

神经网络最重要的就是搭建网络，第一步就是定义网络结构。我这里是创建了一个四层的感知器，参数是根据 MNIST 数据集设定的，网络结构如下：


	# 建立一个四层感知机网络
	class MLP(torch.nn.Module):   # 继承 torch 的 Module
		def __init__(self):
			super(MLP,self).__init__()    # 
			# 初始化三层神经网络 两个全连接的隐藏层，一个输出层
			self.fc1 = torch.nn.Linear(784,512)  # 第一个隐含层  
			self.fc2 = torch.nn.Linear(512,128)  # 第二个隐含层
			self.fc3 = torch.nn.Linear(128,10)   # 输出层
        
		def forward(self,din):
			# 前向传播， 输入值：din, 返回值 dout
			din = din.view(-1,28*28)       # 将一个多行的Tensor,拼接成一行
			dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
			dout = F.relu(self.fc2(dout))
			dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
			# 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
			return dout

网络结构其实很简单，设置了三层 Linear。隐含层激活函数使用 Relu; 输出层使用 Softmax。网上还有其他的结构使用了 droupout，我觉得入门的话有点高级，而且放在这里并没有什么用，搞得很麻烦还不能提高准确率。

### 加载数据集

第二步就是定义全局变量，并加载 MNIST 数据集：

	# 定义全局变量
	n_epochs = 10     # epoch 的数目
	batch_size = 20  # 决定每次读取多少图片
	
	# 定义训练集个测试集，如果找不到数据，就下载
	train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
	test_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
	# 创建加载器
	train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = 0)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = 0)
		

这里参数很多，所以就有很多需要注意的地方了：

* **root** 参数的文件夹即使不存在也没关系，会自动创建

* **transform** 参数，如果不知道要对数据集进行什么变化，这里可自动忽略

* **batch_size** 参数的大小决定了一次训练多少数据，相当于定义了每个 epoch 中反向传播的次数

* **num_workers** 参数默认是 0，即不并行处理数据；我这里设置大于 0 的时候，总是报错，建议设成默认值

如果不理解 epoch 和 batch_size，可以上网查一下资料。（我刚开始学深度学习的时候也是不懂的）

### 训练神经网络

第三步就是训练网络了，代码如下：

	# 训练神经网络
	def train():
		# 定义损失函数和优化器
		lossfunc = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
		# 开始训练
		for epoch in range(n_epochs):
			train_loss = 0.0
			for data,target in train_loader:
				optimizer.zero_grad()   # 清空上一步的残余更新参数值
				output = model(data)    # 得到预测值
				loss = lossfunc(output,target)  # 计算两者的误差
				loss.backward()         # 误差反向传播, 计算参数更新值
				optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
				train_loss += loss.item()*data.size(0)
			train_loss = train_loss / len(train_loader.dataset)
			print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

训练之前要定义损失函数和优化器，这里其实有很多学问，但本文就不讲了，理论太多了。

训练过程就是两层 for 循环：**外层是遍历训练集的次数；内层是每次的批次（batch）**。最后，输出每个 epoch 的 loss。（每次训练的目的是使 loss 函数减小，以达到训练集上更高的准确率）

### 测试神经网络

最后，就是在测试集上进行测试，代码如下：

	# 在数据集上测试神经网络
	def test():
		correct = 0
		total = 0
		with torch.no_grad():  # 训练集中不需要反向传播
			for data in test_loader:
				images, labels = data
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		print('Accuracy of the network on the test images: %d %%' % (
			100 * correct / total))
		return 100.0 * correct / total

这个测试的代码是同学给我的，我觉得这个测试的代码特别好，很简洁，一直用的这个。

代码首先设置 **torch.no_grad()**，定义后面的代码不需要计算梯度，能够节省一些内存空间。然后，对测试集中的每个 batch 进行测试，统计总数和准确数，最后计算准确率并输出。

通常是选择边训练边测试的，这里先就按步骤一步一步来做。

有的测试代码前面要加上 **model.eval()**，表示这是训练状态。但这里不需要，**如果没有 Batch Normalization  和  Dropout 方法，加和不加的效果是一样的**。

# 效果

10 个 epoch 的训练效果，最后能达到大约 85% 的准确率。可以适当增加 epoch，但代码里没有用 gpu 运行，可能会比较慢。

# 参考
写代码的时候，很大程度上参考了下面一些文章，感谢各位作者

1. [基于Pytorch的MLP实现](https://www.jianshu.com/p/65aed5b33cf2)
2. [莫烦 Python ——区分类型 (分类)](https://morvanzhou.github.io/tutorials/machine-learning/torch/3-02-classification/)
3. [使用Pytorch构建MLP模型实现MNIST手写数字识别](https://blog.csdn.net/l1606468155/article/details/89818546) 