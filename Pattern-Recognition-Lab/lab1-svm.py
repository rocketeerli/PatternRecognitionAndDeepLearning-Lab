import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()
# 数据归一化
data = scale(digits.data)
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

print("训练集" + str(X_train.shape))
print("测试集" +str(X_test.shape))

# 创建 SVC 模型
svc_model = svm.SVC(gamma=0.001, C=100, kernel='linear')
# 将训练集应用到 SVC 模型上
svc_model.fit(X_train, y_train)
# 评估模型的预测效果
print(svc_model.score(X_test, y_test))
# 优化参数
svc_model = svm.SVC(gamma=0.001, C=10, kernel='rbf')
svc_model.fit(X_train, y_train)
print(svc_model.score(X_test, y_test))
# 使用创建的 SVC 模型对测试集进行预测
predicted = svc_model.predict(X_test)

X = np.arange(len(y_test))
# 生成比较列表，如果预测的结果正确，则对应位置为0，错误则为1
comp = [0 if y1 == y2 else 1 for y1, y2 in zip(y_test, predicted)]

print("测试集数量：" + str(len(y_test)))
print("错误识别数：" + str(sum(comp)))
print("识别准确率：" + str(1 - float(sum(comp)) / len(y_test)))

# 收集错误识别的样本下标
wrong_index = []
for i, value in enumerate(comp):
    if value: wrong_index.append(i)

# 输出错误识别的样本图像
plt.figure(figsize=(8, 6))
for plot_index, image_index in enumerate(wrong_index):
    image = images_test[image_index]
    plt.subplot(2, 4, plot_index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # 图像说明，8->9 表示正确值为8，被错误地识别成了9
    info = "{right}->{wrong}".format(right=y_test[image_index], wrong=predicted[image_index])
    plt.title(info, fontsize=16)
plt.show()
