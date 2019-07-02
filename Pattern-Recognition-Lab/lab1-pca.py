from os import walk, path  
import numpy as np  
import mahotas as mh  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score  
from sklearn.preprocessing import scale  
from sklearn.decomposition import PCA  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import classification_report  
X = []  
y = []  

# 忽略 警告
import warnings
warnings.filterwarnings("ignore")

#把照片导入Numpy数组，然后把它们的像素矩阵转换成向量：  
for dir_path, dir_names, file_names in walk('orl_faces/'):  
    for fn in file_names:  
        if fn[-3:] == 'pgm':  
            image_filename = path.join(dir_path, fn)  
            X.append(scale(mh.imread(image_filename, as_grey=True).reshape(10304).astype('float32')))  
            y.append(dir_path)  
X = np.array(X)  

#用交叉检验建立训练集和测试集，在训练集上用PCA  
X_train, X_test, y_train, y_test = train_test_split(X, y)  
pca = PCA(n_components=30)   # 80  150 30 20 

# 把所有样本降到150维，然后训练一个逻辑回归分类器。数据集包括40个类；scikit-learn底层会自动用one versus all策略创建二元分类器：  
X_train_reduced = pca.fit_transform(X_train)  
X_test_reduced = pca.transform(X_test)  
print('训练集数据的原始维度是：{}'.format(X_train.shape))  
print('PCA降维后训练集数据是：{}'.format(X_train_reduced.shape))  
classifier = LogisticRegression()  
accuracies = cross_val_score(classifier, X_train_reduced, y_train)  
  
#最后，用交叉验证和测试集评估分类器的性能。分类器的平均综合评价指标（F1 score）是0.88，但是需要花费更多的时间训练，在更多训练实例的应用中可能会更慢。  
  
print('交叉验证准确率是：{}\n{}'.format(np.mean(accuracies), accuracies))  
classifier.fit(X_train_reduced, y_train)  
predictions = classifier.predict(X_test_reduced)  
print(classification_report(y_test, predictions))  
