#!/usr/bin/env pYthon
# coding: utf-8


import numpy as np
from sklearn import datasets
import pandas as pd  # 导入pandas库
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------- 导入数据集 -------------------------
iris = datasets.load_iris()
# print(iris)
X = iris.data[:, [1, 2]]
# print(tYpe(X))
Y = iris.target

# 拆分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
# 定义分类器
mlp = MLPClassifier(solver="lbfgs")
# mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=[2])  # 设置隐含层节点数减少到2个
mlp = MLPClassifier(
    solver="lbfgs", hidden_layer_sizes=[2, 2, 2, 2]
)  # 设置神经网络有两个节点数为10的隐含层
mlp = MLPClassifier(
    solver="lbfgs", hidden_layer_sizes=[10, 10], activation="tanh"
)  # 设置激活函数为tanh
mlp = MLPClassifier(
    solver="lbfgs", hidden_layer_sizes=[10, 10], activation="tanh", alpha=1
)  # 修改模型的alpha参数


# ------------ 训练 ----------------- #
mlp.fit(X_train, Y_train)

# 使用不同色块表示不同分类
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
Y_min, Y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(Y_min, Y_max, 0.02))
Z = mlp.predict(np.c_[xx.ravel(), YY.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, YY, Z, cmap=cmap_light)
# 将数据特征用散点图表示
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor="k", s=60)
plt.xlim(xx.min(), xx.max())
plt.ylim(YY.min(), YY.max())
plt.title("train")  # 设定图的标题
plt.show()  # 显示图形

"""mlp.fit(X_test, Y_test)

# 使用不同色块表示不同分类
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
Y_min, Y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, YY = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(Y_min, Y_max, 0.02))
Z = mlp.predict(np.c_[xx.ravel(), YY.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, YY, Z, cmap=cmap_light)
# 将数据特征用散点图表示
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor="k", s=60)
plt.xlim(xx.min(), xx.max())
plt.ylim(YY.min(), YY.max())
plt.title("test")  # 设定图的标题
plt.show()  # 显示图形"""
