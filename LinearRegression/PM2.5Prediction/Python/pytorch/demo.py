#!/usr/bin/env python3

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append("..")

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as Data
from data_processing import (load_data, normalization, split_train_valid_set,
                             transform_data)
from torch import nn
from torch.nn import init

# 1. 导入数据
data = load_data()
train_data, valid_data = split_train_valid_set(data)
train_set = transform_data(train_data, data_path="../../Data/train_set.csv")
valid_set = transform_data(train_data, data_path="../../Data/valid_set.csv")
train_X = torch.tensor(
    normalization(train_set[train_set.columns[:-1]].to_numpy()), dtype=torch.float
)
train_Y = torch.tensor(
    train_set[train_set.columns[-1]].to_numpy().reshape(-1, 1), dtype=torch.float
)
valid_X = torch.tensor(
    normalization(valid_set[valid_set.columns[:-1]].to_numpy()), dtype=torch.float
)
valid_Y = torch.tensor(
    valid_set[valid_set.columns[-1]].to_numpy().reshape(-1, 1), dtype=torch.float
)

# 2. 数据播放
batch_size = 128
# 结合 feature 张量与 label 向量
dataset = Data.TensorDataset(train_X, train_Y)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=False)

# 3. 定义模型
# 3.1 方法一： 继承重写 nn.Module
class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, X):
        y = self.linear(X)
        return y


# 3.2 方法二：使用 nn.Sequential
# 写法一
# net = nn.Sequential(
#     nn.Linear(train_X.shape[1], 1)
#     # ... 可添加其他层
# )
# # 写法二
# net = nn.Sequential()
# net.add_module("linear", nn.Linear(train_X.shape[1], 1))
#
# # 写法三
# from collections import OrderedDict
#
# net = nn.Sequential(OrderedDict([("linear", nn.Linear(train_X.shape[1], 1))]))

net = LinearNet(train_X.shape[1])

# 查看模型参数
for param in net.parameters():
    print(param)

# 4. 初始化模型参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=20)
# init.normal_(net[0].weight, mean=0, std=0.01)
# init.constant_(net[0].bias, val=0)

# 5. 定义损失函数
loss = nn.MSELoss()

# 6. 优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
# 还可以对多层子网络分别设置学习率
# optimizer = optim.SGD(
#     [
#         {"params": net.subnet1.parameters()},  # lr = 0.03
#         {"params": net.subnet2.parameters(), "lr": 0.01},
#     ]
# )
# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group["lr"] *= 0.1  # 学习率为之前的 0.1 倍

# 7. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于 net.zero_grad()
        l.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print("Epoch {} with loss {}".format(epoch, loss(net(train_X), train_Y).item()))
