#!/usr/bin/env python3
import sys

sys.path.append("..")
from data_processing import load_data, split_train_valid_set, transform_data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    reg = LinearRegression()
    scalar = StandardScaler()
    data = load_data("../../Data/train.csv")
    train_data, valid_data = split_train_valid_set(data)
    train_set = transform_data(train_data, data_path="../../Data/train_set.csv")
    valid_set = transform_data(train_data, data_path="../../Data/valid_set.csv")
    train_X = train_set[train_set.columns[:-1]].to_numpy()
    train_Y = train_set[train_set.columns[-1]].to_numpy()
    scalar = StandardScaler()
    scalar.fit(train_X)
    normed_train_X = scalar.transform(train_X)
    reg.fit(normed_train_X, train_Y)

    valid_X = valid_set[valid_set.columns[:-1]].to_numpy()
    valid_Y = valid_set[valid_set.columns[-1]].to_numpy()
    scalar.fit(valid_X)
    normed_valid_X = scalar.transform(valid_X)
    pred_valid = reg.predict(normed_valid_X)
    print(
        "Mean squared error: %.2f"
        % mean_squared_error(pred_valid, valid_set[valid_set.columns[-1]])
    )
    print(
        "Coefficient of determination: %.2f"
        % r2_score(valid_set[valid_set.columns[-1]], pred_valid)
    )
