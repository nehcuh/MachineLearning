#!/usr/bin/env python3

"""
数据预处理

1. 数据导入与预处理
2. 数据集分割
3. 数据格式转换
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

TRAIN_DATA = "../../Data/train.csv"


def load_data(data_path: str = TRAIN_DATA) -> pd.DataFrame:
    """数据导入与预处理

    Args:
        data_path (str, optional): 数据路径

    Returns:
        pd.DataFrame: 以日期为索引，特征为列的透视表
    """
    columns = ["date", "addr", "item"] + list(map(str, range(24)))
    df = pd.read_csv(data_path, skiprows=1, names=columns, encoding="big5")
    df = df.drop(columns="addr").melt(
        id_vars=["date", "item"], var_name="hour", value_name="value"
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["hour"] + ":00:00")
    df = df.set_index("datetime").drop(columns=["date", "hour"])
    df.loc[df["value"] == "NR", "value"] = 0.0
    df["value"] = df["value"].astype(float)
    return df.pivot_table(
        index="datetime", columns="item", values="value", aggfunc="sum"
    )


def split_train_valid_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """数据集分割为测试集与验证集

    Args:
        df (pd.DataFrame): 以日期为索引，特征为列的透视表

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 测试集与验证集 Tuple
    """
    train_set = df.loc[df.index.month != 12]
    valid_set = df.loc[df.index.month == 12]
    return (train_set, valid_set)


def transform_data(
    df: pd.DataFrame, if_persistent: bool = True, data_path: str = None
) -> pd.DataFrame:
    """数据格式转换为特征与标签

    Args:
        df (pd.DataFrame): 以日期为索引，特征为列的透视表
        if_persistent (bool, optional): 是否进行数据持久化
        data_path (str, optional): 当 if_persistent 为 True 时，需要设置 data_path, 默认为 None

    Returns:
        pd.DataFrame: 特征列与标签列整合的数据集
    """
    if if_persistent:
        try:
            df = pd.read_csv(data_path)
            return df
        except:
            print("No local data exists!")
    results = dict()
    param_list = df.columns.tolist()
    for param in param_list:
        for i in range(9):
            results["{:2d}_{}".format(i + 1, param)] = []
    results["10_PM2.5"] = []
    for timestamp in df.index:
        start = timestamp
        end = timestamp + pd.Timedelta(hours=9)  # 计时从 0 点开始
        sub_df = df.loc[start:end]
        if len(sub_df) == 10:  # 保证数据对齐
            results["10_PM2.5"].append(sub_df.iloc[-1]["PM2.5"])
            feature_df = sub_df.iloc[:-1]
            for i in range(9):
                for param in param_list:
                    results["{:2d}_{}".format(i + 1, param)].append(
                        feature_df.iloc[i][param]
                    )
    if if_persistent:
        if not data_path:
            raise ValueError("Param data_path should be specified.")
    results = pd.DataFrame(results)
    results.to_csv(data_path, index=False)
    return results


def normalization(
    df: Union[pd.DataFrame, np.array], method: str = "standard"
) -> Union[pd.DataFrame, np.array]:
    """
    标准化

    Args:
        df (Union[pd.DataFrame, np.array]): 原始数据集
        method (str, optional): 归一化方法，支持 "standard" (x - mean) / std, "maxmin" (x - min) / (max - min)

    Returns:
        Union[np.array, pd.DataFrame]: 归一化之后的数据集
    """
    if method == "standard":
        return (df - df.mean(axis=0)) / df.std(axis=0)
    elif method == "maxmin":
        return (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    else:
        raise ValueError("Not Implemented Method {}".format(method))
