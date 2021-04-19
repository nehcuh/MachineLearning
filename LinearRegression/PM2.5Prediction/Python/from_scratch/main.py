#!/usr/bin/env python3

import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np

from data_processing import load_data, split_train_valid_set, transform_data
from from_scratch.data_processing import normalization
from linear_model import (adaptive_gradient_descent, gradient_descent,
                          pseudo_inverse, stochastic_gradient_descent)

if __name__ == "__main__":
    df = load_data("../../Data/train.csv")
    train_df, valid_df = split_train_valid_set(df)
    train_set = transform_data(train_df, data_path="../../Data/train_set.csv")
    valid_set = transform_data(valid_df, data_path="../../Data/valid_set.csv")

    train_X = train_set[train_set.columns[:-1]].to_numpy()
    train_Y = train_set[train_set.columns[-1]].to_numpy()

    valid_X = valid_set[valid_set.columns[:-1]].to_numpy()
    valid_Y = valid_set[valid_set.columns[-1]].to_numpy()

    # Pseudo Inverse
    train_loss, general_weights = pseudo_inverse(train_X, train_Y, lambdaL2=0)
    bias = general_weights[0]
    weights = general_weights[1:]
    print("Pesudo Inverse with train cost: {}".format(train_loss))
    ## loss for validation set
    valid_pred_Y = valid_X.dot(weights) + bias
    valid_loss = (
        (valid_pred_Y - valid_Y.reshape(-1, 1)).T.dot(
            valid_pred_Y - valid_Y.reshape(-1, 1)
        )
        / len(valid_Y)
        / 2
    )
    print("Pesudo Inverse with valid cost: {}".format(valid_loss))
    print("weights and bias are: ", weights[:3], bias)

    normed_train_X = normalization(train_X)
    normed_valid_X = normalization(valid_X)
    train_loss, general_weights = pseudo_inverse(normed_train_X, train_Y, lambdaL2=0)
    bias = general_weights[0]
    weights = general_weights[1:]
    print("Pesudo Inverse with train cost: {}".format(train_loss))
    ## loss for validation set
    valid_pred_Y = normed_valid_X.dot(weights) + bias
    valid_loss = (
        (valid_pred_Y - valid_Y.reshape(-1, 1)).T.dot(
            valid_pred_Y - valid_Y.reshape(-1, 1)
        )
        / len(valid_Y)
        / 2
    )
    print("Pesudo Inverse with normalization has valid cost: {}".format(valid_loss))
    print("weights and bias are: ", weights[:3], bias)

    # AdaGrad
    losses, general_weights = adaptive_gradient_descent(normed_train_X, train_Y)
    bias = general_weights[0]
    weights = general_weights[1:]
    valid_pred_Y = normed_valid_X.dot(weights) + bias
    valid_loss = (
        (valid_pred_Y - valid_Y.reshape(-1, 1)).T.dot(
            valid_pred_Y - valid_Y.reshape(-1, 1)
        )
        / len(valid_Y)
        / 2
    )
    print("Adaptive Gradient Descent with valid cost: {}".format(valid_loss))
    print("weights and bias are: ", weights[:3], bias)

    # GD and SGD method can't converge
    # Normalization saves it
    weights = np.random.randn(normed_train_X.shape[1])
    bias = np.array([20])
    W = np.append(bias, weights)
    losses, general_weights = gradient_descent(
        normed_train_X, train_Y, W=W, num_epochs=10000
    )
    weights = general_weights[1:]
    bias = general_weights[0]
    valid_pred_Y = normed_valid_X.dot(weights) + bias
    valid_loss = (
        (valid_pred_Y - valid_Y.reshape(-1, 1)).T.dot(
            valid_pred_Y - valid_Y.reshape(-1, 1)
        )
        / len(valid_Y)
        / 2
    )
    print("Gradient Descent with valid cost: {}".format(valid_loss))
    print("weights and bias are: ", weights[:3], bias)

    losses, general_weights = stochastic_gradient_descent(
        normed_train_X, train_Y, batch_size=128
    )
    weights = general_weights[1:]
    normed_valid_X = normalization(valid_X)
    bias = general_weights[0]
    valid_pred_Y = normed_valid_X.dot(weights) + bias
    valid_loss = (
        (valid_pred_Y - valid_Y.reshape(-1, 1)).T.dot(
            valid_pred_Y - valid_Y.reshape(-1, 1)
        )
        / len(valid_Y)
        / 2
    )
    print("Stochastic Gradient Descent with valid cost: {}".format(valid_loss))
