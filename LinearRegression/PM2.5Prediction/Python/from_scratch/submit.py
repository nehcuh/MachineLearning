#!/usr/bin/env python3

"""
作业提交
"""
import numpy as np
import pandas as pd

from data_processing import load_data, transform_data
from linear_model import pseudo_inverse

# 测试数据预处理
df_test = pd.read_csv(
    "../../Data/test.csv",
    header=None,
    names=["id", "item"] + list(map(str, range(1, 10))),
)
df_test = df_test.melt(id_vars=["id", "item"], var_name="hour", value_name="value")
df_test.loc[df_test["value"] == "NR", "value"] = 0.0
df_test["value"] = df_test["value"].astype("float")
df_test = df_test.set_index(["id", "item", "hour"]).unstack(level=-1).unstack(level=-1)
df_test.columns = df_test.columns.droplevel(level=0)
df_test.columns = df_test.columns.map(lambda x: "{:2d}_{}".format(int(x[0]), x[1]))
df_test = df_test.reindex(
    df_test.index.to_series().str.split("_").str[1].astype(int).sort_values().index
)

# 导入原始数据
df_train = load_data("../../Data/train.csv")

# 原始数据格式转换
df_transformed = transform_data(df_train, data_path="../../Data/formated_train.csv")

# 使用伪逆求解参数
W = pseudo_inverse(
    df_transformed[df_transformed.columns[:-1]],
    df_transformed[df_transformed.columns[-1]],
    lambdaL2=0.0,
)

# 将测试数据排列为与训练数据一致
feature_test = df_test[df_transformed.columns[:-1]]
feature_test.loc[:, "value"] = np.dot(feature_test, W[1][1:]) + W[1][0]
feature_test[["value"]].to_csv("../../Data/submit.csv")
