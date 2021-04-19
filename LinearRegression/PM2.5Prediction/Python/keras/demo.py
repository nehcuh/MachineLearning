#!/usr/bin/env python3

import os
import sys

sys.path.append("..")
import numpy as np

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import random
from typing import Tuple, Union

import pandas as pd
from data_processing import (load_data, normalization, split_train_valid_set,
                             transform_data)

import keras.applications as kapp
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD

# 1. 数据导入
data = load_data()
train_data, valid_data = split_train_valid_set(data)
train_set = transform_data(train_data, data_path="../../Data/train_set.csv")
valid_set = transform_data(valid_data, data_path="../../Data/valid_set.csv")
train_X = normalization(train_set[train_set.columns[:-1]].to_numpy())
train_Y = train_set[train_set.columns[-1]].to_numpy().reshape(-1, 1)
valid_X = normalization(valid_set[train_set.columns[:-1]].to_numpy())
valid_Y = valid_set[valid_set.columns[-1]].to_numpy().reshape(-1, 1)

# 2. 数据播放
def data_iter(
    X: Union[pd.DataFrame, np.array],
    Y: Union[pd.Series, np.array],
    batch_size: int = 10,
    if_shuffle=True,
) -> Tuple[np.array, np.array]:
    """数据播放

    Args:
        X (pd.DtaFrame, np.array): features 张量
        Y (pd.Series, np.array): labels 向量
        batch_size (int, optional): 批量大小, 默认为 10
        if_shuffle (bool, optional): 是否 shuffle，默认为 True

    Returns:
        Tuple[np.array, np.array]: 批量 features 和批量 labels
    """
    if len(Y.shape) == 1:
        Y = np.array(Y).reshape(-1, 1)
    X = np.array(X)
    num_examples = len(X)
    indices = list(range(num_examples))
    if if_shuffle:
        random.shuffle(indices)
    for batch in range(0, num_examples, batch_size):
        j = indices[batch : min(batch + batch_size, num_examples)]
        yield (X[j], Y[j])


# 3. 模型定义
model = Sequential(Dense(1, input_dim=train_X.shape[1]))

# 4. 优化器
model.compile(optimizer="rmsprop", loss="mse")

# 5. 训练
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd)
model.fit(train_X, train_Y, batch_size=32, epochs=100)
score = model.evaluate(valid_X, valid_Y, batch_size=32)
