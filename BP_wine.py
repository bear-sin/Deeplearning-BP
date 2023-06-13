#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# --- 导入数据 --- #
wine = load_wine()
# --- 构造特征变量与目标变量 --- #
X = wine.data[:, :2]
y = wine.target
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
# 定义分类器
mlp = MLPClassifier(solver='lbfgs')
# mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10])  # 设置隐含层节点数减少到10个
# mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10, 10])  # 设置神经网络有两个节点数为10的隐含层
# mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10, 10], activation='tanh')  # 设置激活函数为tanh
# mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10, 10], activation='tanh',alpha=1) # 修改模型的alpha参数


# ------------ 训练 ----------------- #
mlp.fit(X_train, y_train)

# 使用不同色块表示不同分类
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 将数据特征用散点图表示
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=60)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("MLPClassifier:solver=lbfgs")  # 设定图的标题
plt.show()  # 显示图形

