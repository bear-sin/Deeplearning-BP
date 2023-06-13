#!/usr/bin/env python
# coding: utf-8


from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


#加载MNIST手写数字数据集
mnist = fetch_openml('mnist_784')
#mnist = fetch_mldata('MNIST original')
#mnist = fetch_openml('MNIST original’)
print('样本数量：{}, 样本特征数：{}'.format(mnist.data.shape[0], mnist.data.shape[1]))


# 数据预处理--归一化
X = mnist.data/255.
y = mnist.target
# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 5000, test_size=1000,random_state=62)

# 设置神经网络有两个100个节点的隐含层
mlp_hw = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[100,100], activation='relu', alpha = 1e-5,random_state=62)
# 训练神经网络模型
mlp_hw.fit(X_train,y_train)
# 打印模型得分
print('测试数据集得分：{:.2f}%'.format(mlp_hw.score(X_test,y_test)*100))


# ---------- 自己输入图像做测试 ------------ #
# from PIL import Image
# image=Image.open('0.png').convert('F')
# image=image.resize((28,28))
# image.show()
#
# arr=[]
# for i in range(28):
#     for j in range(28):
#         pixel = 1.0 - float(image.getpixel((j,i)))/255.
#         #pixel = float(image.getpixel((j,i)))/255.
#
#         arr.append(pixel)
# arr1 = np.array(arr).reshape(1,-1)
# print('图片中的数字是:{0}'.format(mlp_hw.predict(arr1)[0]))
