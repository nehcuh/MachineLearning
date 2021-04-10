#!/usr/bin/env python3

"""
线性回归从零开始
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

from mxnet import autograd, nd

# 1 . 数据集生成
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1]
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 2. 数据可视化
def use_svg_display():
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy())

# 3. 数据播放
def data_iter(features, labels, batch_size, if_shuffle: bool = True):
    num_examples = len(features)
    indices = list(range(num_examples))
    if if_shuffle:
        random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array([i, min((i + 1) * batch_size, num_examples)])
        yield features.take(j), labels.take(j)


# 4. 初始化参数模型
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

batch_size = 10

# 5. 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b


# 6. 定义损失函数
def square_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 7. 优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] -= lr * param.grad / batch_size


# 8. 模型训练
lr = 0.03
num_epochs = 30
net = linreg
loss = square_loss
for epoch in range(num_epochs):
    for X, y in data_iter(features, labels, batch_size):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    if epoch % 10 == 0:
        print("epoch {}, loss {}".format(epoch, train_l.mean().asnumpy()))
