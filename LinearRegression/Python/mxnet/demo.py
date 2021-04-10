#!/usr/bin/env python3

import sys

sys.path.append("..")

from data_processing import (load_data, normalization, split_train_valid_set,
                             transform_data)

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn

# 1. 数据导入
data = load_data("../../Data/train.csv")
train_data, valid_data = split_train_valid_set(data)
train_set = transform_data(train_data, data_path="../../Data/train_set.csv")
valid_set = transform_data(valid_data, data_path="../../Data/valid_set.csv")
features = nd.array(
    normalization(train_set[train_set.columns[:-1]].to_numpy()), dtype="float32"
)
labels = nd.array(train_set[train_set.columns[-1]].to_numpy(), dtype="float32")

# 2. 数据播放
batch_size = 128
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=False)

# 3. 定义模型
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.1))

# 4. 优化方式
# trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": 0.2})
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.02})

# 5. 损失函数
loss = gloss.L2Loss()

# 6. 训练
num_epochs = 1000
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    if epoch % 100 == 0:
        print("Epoch {} with loss {}".format(epoch, l.mean().asnumpy()))

dense = net[0]
print("weights: ", dense.weight.data())
print("bias: ", dense.bias.data())

valid_loss = loss(
    net(nd.array(normalization(valid_set[valid_set.columns[:-1]]))),
    nd.array(valid_set[valid_set.columns[-1]]),
)
print("valid loss: ", valid_loss)
