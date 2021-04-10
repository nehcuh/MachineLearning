#!/usr/bin/env python3

"""
利用 Pytorch 实现从零开始线性回归

1. 数据播放
2. 模型设置
3. 损失函数
4. 优化方式
"""

import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display

# 1. 数据集生成
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1]
labels += torch.tensor(
    np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32
)

# 2. 数据可视化
def use_svg_display():
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy())


# 3. 数据播放
def data_iter(batch_size, features, labels, if_shuffle: bool = True):
    num_examples = len(features)
    indices = list(range(num_examples))
    if if_shuffle:
        random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i : min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 4. 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

batch_size = 10

# 5. 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 6. 定义损失函数
def square_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 7. 优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 8. 模型训练
lr = 0.03
num_epochs = 30
net = linreg
loss = square_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels, if_shuffle=False):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    if epoch % 10 == 0:
        print("epoch {}, loss {}".format(epoch, train_l.mean().item()))
print("weights is", w)
print("bias is", b)
