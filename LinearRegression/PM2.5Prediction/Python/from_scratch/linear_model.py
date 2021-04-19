#!/usr/bin/env python3
"""
线性模型 (L2 正则)

1. 伪逆矩阵
2. 梯度下降
3. 随机梯度下降
4. AdamGrad
"""
import math
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def pseudo_inverse(
    X: Union[np.array, pd.DataFrame], Y: Union[np.array, pd.Series], lambdaL2: float
) -> Tuple[float, np.array]:
    """伪逆矩阵求解权重向量
    1. $argmin_W [(XW - Y)^T(XW - Y) / 2m + \lambda ||W||^2]$
    2. f = (XW - Y)^T(XW - Y) / 2m + \lambda ||W||^2
    3. $\frac{\partial f}{\partial W} = 0$ => $X^T(XW - Y) / m + \lambda W = 0$
    4. W = [(X^TX) / m + \lambda]^{-1}X^TY

    Args:
        X (Union[np.array, pd.DataFrame]): 特征张量
        Y (Union[np.array, pd.Series]): 标签向量

    Returns:
        Tuple[float, np.array]: cost, 权重向量
    """
    # 1. 数据格式调整
    if len(Y.shape) == 1:
        Y = np.array(Y).reshape(-1, 1)
    # 2. 将 bias 合并到 features 张量中
    general_X = np.hstack([np.ones(len(X)).reshape(-1, 1), np.array(X)])
    # 3. 计算广义 W 向量: W[0] 为 bias, W[1:] 为权重向量
    W = np.dot(
        np.dot(
            np.linalg.pinv(
                np.dot(general_X.T, general_X) + lambdaL2 * np.eye(general_X.shape[1])
            ),
            general_X.T,
        ),
        Y,
    )
    # 4. cost 计算
    cost = np.dot((np.dot(general_X, W) - Y).T, (np.dot(general_X, W) - Y)) / 2 / len(
        X
    ) - lambdaL2 * np.dot(W.T, W)
    return cost.ravel()[0], W


def gradient_descent(
    X: Union[np.array, pd.DataFrame],
    Y: Union[np.array, pd.Series],
    W: Union[np.array, pd.Series] = None,
    lambdaL2: float = 0.0,
    learning_rate: float = 0.002,
    num_epochs: int = 50000,
    eps: float = 1e-5,
) -> Tuple[List[float], np.array]:
    """梯度下降

    Args:
        X (Union[np.array, pd.DataFrame]): 特征张量
        Y (Union[np.array, pd.Series]): 标签向量
        W (Union[np.array, pd.Series], optional): 初始权重, 默认为 None
        lambdaL2 (float): L2 正则惩罚系数, 默认为 0
        learning_rate (float, optional): 学习率，默认为 0.002
        eps (float, optional): 收敛域, 默认为 1E-5

    Returns:
        Tuple[List[float], np.array]: 每次迭代过程的 cost 和权重向量
    """
    if len(Y.shape) == 1:
        Y = np.array(Y).reshape(-1, 1)
    costs = list()
    general_X = np.hstack([np.ones(len(X)).reshape(-1, 1), X])
    if W is None:  # 初始权重随机生成，W[0] = bias
        # W = np.vstack([np.array([1]), np.random.randn(X.shape[1], 1)])
        W = np.random.randn(general_X.shape[1]).reshape(-1, 1)
    if len(W.shape) == 1:
        W = np.array(W).reshape(-1, 1)
    for epoch in range(num_epochs):
        cost = (
            (general_X.dot(W) - Y).T.dot(general_X.dot(W) - Y) / 2 / len(X)
            - lambdaL2 * W.T.dot(W)
        ).ravel()[0]
        costs.append(cost)
        if cost < eps:
            return costs, W
        G = general_X.T.dot(general_X.dot(W) - Y) / len(X) + lambdaL2 * W
        W = W - learning_rate * G
        if epoch % 1000 == 0:
            print("current epoch is {} with cost {}".format(epoch, cost))
            print("current bias: ", W[0])
            print(
                "current prediction is: ",
                general_X.dot(W),
                "\n while true value is:",
                Y,
            )
    return costs, W


def stochastic_gradient_descent(
    X: Union[np.array, pd.DataFrame],
    Y: Union[np.array, pd.Series],
    batch_size: int,
    W: Union[np.array, pd.Series] = None,
    lambdaL2: float = 0.0,
    learning_rate: float = 0.002,
    num_epochs: int = 50000,
    eps: float = 1e-5,
) -> Tuple[List[float], np.array]:
    """梯度下降

    Args:
        X (Union[np.array, pd.DataFrame]): 特征张量
        Y (Union[np.array, pd.Series]): 标签向量
        batch_size (int): 批量大小
        W (Union[np.array, pd.Series], optional): 初始权重, 默认为 None
        lambdaL2 (float): L2 正则惩罚系数, 默认为 0
        learning_rate (float, optional): 学习率，默认为 0.002
        eps (float, optional): 收敛域, 默认为 1E-5

    Returns:
        Tuple[List[float], np.array]: 每次迭代过程的 cost 和权重向量
    """
    costs = []
    if len(Y.shape) == 1:
        Y = np.array(Y).reshape(-1, 1)
    general_X = np.hstack([np.ones(len(X)).reshape(-1, 1), X])
    if W is None:
        W = np.random.randn(general_X.shape[1]).reshape(-1, 1)
    if len(W.shape) == 1:
        W = np.array(W).reshape(-1, 1)
    idxes = np.arange(len(X))
    random.shuffle(idxes)
    general_X = general_X[idxes]
    Y = Y[idxes]
    num_batch = math.ceil(len(X) / batch_size)
    for epoch in range(num_epochs):
        for batch in range(num_batch):
            sub_X = general_X[batch : min((batch + 1) * batch_size, general_X.shape[0])]
            sub_Y = Y[batch : min((batch + 1) * batch_size, Y.shape[0])]
            cost = (
                (sub_X.dot(W) - sub_Y).T.dot(sub_X.dot(W) - sub_Y) / 2 / len(sub_X)
                - lambdaL2 * W.T.dot(W)
            ).ravel()[0]
            costs.append(cost)
            if cost < eps:
                return costs, W
            G = sub_X.T.dot(sub_X.dot(W) - sub_Y) / len(X) + lambdaL2 * W
            W = W - learning_rate * G
        if epoch % 1000 == 0:
            print("current epoch is {} with cost {}".format(epoch, cost))
    return costs, W


def adaptive_gradient_descent(
    X: Union[np.array, pd.DataFrame],
    Y: Union[np.array, pd.Series],
    W: Union[np.array, pd.Series] = None,
    lambdaL2: float = 0.0,
    learning_rate: float = 2,
    num_epochs: int = 50000,
    eps: float = 1e-5,
) -> Tuple[List[float], np.array]:
    """梯度下降

    Args:
        X (Union[np.array, pd.DataFrame]): 特征张量
        Y (Union[np.array, pd.Series]): 标签向量
        W (Union[np.array, pd.Series], optional): 初始权重, 默认为 None
        lambdaL2 (float): L2 正则惩罚系数, 默认为 0
        learning_rate (float, optional): 学习率，默认为 0.002
        eps (float, optional): 收敛域, 默认为 1E-5

    Returns:
        Tuple[List[float], np.array]: 每次迭代过程的 cost 和权重向量
    """
    if len(Y.shape) == 1:
        Y = np.array(Y).reshape(-1, 1)
    costs = list()
    general_X = np.hstack([np.ones(len(X)).reshape(-1, 1), X])
    if not W:  # 初始权重随机生成，W[0] = bias
        # W = np.vstack([np.array([1]), np.random.randn(X.shape[1], 1)])
        W = np.random.randn(general_X.shape[1]).reshape(-1, 1)
    if len(W.shape) == 1:
        W = np.array(W).reshape(-1, 1)
    s_grad = np.zeros((general_X.shape[1], 1))
    for epoch in range(num_epochs):
        cost = (
            (general_X.dot(W) - Y).T.dot(general_X.dot(W) - Y) / 2 / len(X)
            - lambdaL2 * W.T.dot(W)
        ).ravel()[0]
        costs.append(cost)
        if cost < eps:
            return costs, W
        G = general_X.T.dot(general_X.dot(W) - Y) / len(X) + lambdaL2 * W
        s_grad += G ** 2
        ada_grad = np.sqrt(s_grad)
        W = W - learning_rate * G / ada_grad
        if epoch % 1000 == 0:
            print("current epoch is {} with cost {}".format(epoch, cost))
    return costs, W
