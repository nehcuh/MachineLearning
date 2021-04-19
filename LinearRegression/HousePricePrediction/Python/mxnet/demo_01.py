#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss

# 1. 导入数据
df_train = pd.read_csv("train.csv", index_col="Id")
df_test = pd.read_csv("test.csv", index_col="Id")

all_features = pd.concat([df_train.iloc[:, :-1], df_test], axis=0)

# 2. 数据预处理
# 对数值型 feature 进行处理
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index.tolist()

# 标准化 features，这里只使用 train dataset, 避免混乱 test set 信息
means = np.expand_dims(
    np.apply_along_axis(
        lambda x: x[~np.isnan(x)].mean(), axis=0, arr=df_train[numeric_features]
    ),
    axis=0,
)
stds = np.expand_dims(
    np.apply_along_axis(
        lambda x: x[~np.isnan(x)].std(), axis=0, arr=df_train[numeric_features]
    ),
    axis=0,
)
all_features[numeric_features] = (all_features[numeric_features] - means) / stds

all_features[numeric_features] = all_features[numeric_features].fillna(
    0.0
)  # 标准化后，均值为 0

# all_features[numeric_features] = all_features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std())
# )
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
# all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 哑变量设置
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = df_train.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(df_train.SalePrice.values).reshape((-1, 1))

# 3. 模型设置 (线性模型)


def get_net():
    net = gluon.nn.Sequential()
    net.add(gluon.nn.Dense(1))
    net.initialize()
    return net


# 4. 定义损失函数
loss = gloss.L2Loss()

# 5. 模型评价 (RMSE)


def log_rmse(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float("inf"))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


# 6. 训练
def train(
    net,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(
        gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True
    )
    # 使用 Adam 优化算法
    trainer = gluon.Trainer(
        net.collect_params(),
        "adam",
        {"learning_rate": learning_rate, "wd": weight_decay},
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 7. k 折交叉验证
def get_k_fold(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams["figure.figsize"] = figsize


def semilogy(
    x_vals,
    y_vals,
    x_label,
    y_label,
    x2_vals=None,
    y2_vals=None,
    legend=None,
    figsize=(3.5, 2.5),
):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=":")
        plt.legend(legend)


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0.0, 0.0
    for i in range(k):
        data = get_k_fold(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size
        )
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(
                range(1, num_epochs + 1),
                train_ls,
                "epochs",
                "rmse",
                range(1, num_epochs + 1),
                valid_ls,
                ["train", "valid"],
            )
        print(
            "fold {}, train rmse {}, valid rmse {}".format(
                i, train_ls[-1], valid_ls[-1]
            )
        )
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(
    k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size
)
print(
    "%d-fold validation: avg train rmse %f, avg valid rmse %f" % (k, train_l, valid_l)
)
