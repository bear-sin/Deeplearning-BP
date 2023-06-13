#!/usr/bin/env python
# coding: utf-8


# === 实现异或 === #


import numpy as np
import matplotlib.pyplot as plt

err_goal = 1e-6  # 设置误差
max_epoch = 10000  # 最大训练次数
lr = 0.1  # 学习速率
a = 0.5  # 惯性系数
Oi = 0
Ok = 0  # 隐含层和输出层赋初值
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
T = np.array([0, 1, 1, 0])  # 设置训练集

# 初始化各参数
M = 2
q = 8
L = 1
N = 4
Wij = np.random.uniform(-1, 1, (q, M))
Wki = np.random.uniform(-1, 1, (L, q))
Wij0 = np.zeros(Wij.shape)
Wki0 = np.zeros(Wki.shape)
b1 = np.random.uniform(-1, 1, (q, 1))
b2 = np.random.uniform(-1, 1, (L, 1))


# 前馈计算，求系统输出及误差；输出层为线性函数
for epoch in range(max_epoch):
    for p in range(N):
        NETi = np.dot(Wij, np.expand_dims(X[:, p], axis=1)) + b1
        # for i in range(q):
        #     Oi[i] = 1/(1+np.exp(-NETi[i]))
        Oi = 1 / (1 + np.exp(-NETi))

        Ok = np.dot(Wki, Oi) + b2
        E = (np.dot((T[p] - Ok).T, T[p] - Ok)) / 2

        if E < err_goal:
            break

        # 反向传播求网络权值及偏置修正量
        deltak = T[p] - Ok
        W = Wki
        Wki = Wki + lr * deltak * Oi.T + a * (Wki - Wki0)
        Wki0 = W
        b2 = b2 + lr * deltak

        deltai = Oi * (1 - Oi) * (np.dot(Wki.T, deltak))
        W = Wij
        Wij = Wij + lr * np.dot(deltai, np.expand_dims(X[:, p], axis=1).T) + a * (Wij - Wij0)
        Wij0 = W
        b1 = b1 + lr * deltai


# 在(-1,1)~(2,2)之间形成点阵
x = np.arange(-1, 2, 0.1)
N = len(x)
xx = np.zeros((2, N * N))
for i in range(N):
    for j in range(N):
        xx[0, (i - 1) * N + j] = x[i]
        xx[1, (i - 1) * N + j] = x[j]

y = np.zeros(xx.shape[1])

for k in range(xx.shape[1]):
    Xall = xx[:, k]
    NETi = np.dot(Wij, np.expand_dims(Xall, axis=1)) + b1  # 检测输出
    Oi = 1 / (1 + np.exp(-NETi))
    Ok = np.dot(Wki, Oi) + b2
    y[k] = Ok  # 保存点阵中每个输出

# print(y)


# 以0.5为界，将输出标位两类
plt.figure()
plt.scatter(xx[0, y >= 0.5], xx[1, y >= 0.5], s=5, c='r', marker='o', label='First category')
plt.scatter(xx[0][y < 0.5], xx[1, y < 0.5], s=5, c='b', marker='*', label='Second category')

# 训练样本标粗
plt.scatter(X[0, T >= 0.5], X[1, T >= 0.5], s=50, c='r', marker='o')
plt.scatter(X[0, T < 0.5], X[1, T < 0.5], s=50, c='b', marker='*')

plt.legend()
plt.title('The classification results')
plt.show()

