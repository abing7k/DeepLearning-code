import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l  # 假设已安装d2l库，用于绘图等实用工具
import matplotlib.pyplot as plt  # 用于可视化，确保matplotlib已安装

# 数据集下载和缓存相关常量
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):
    """
    下载一个DATA_HUB中的文件，返回本地文件名。
    如果文件已存在且SHA-1匹配，则使用缓存文件。
    :param name: 数据集名称
    :param cache_dir: 缓存目录
    :return: 本地文件名
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """
    下载并解压zip/tar文件。
    :param name: 数据集名称
    :param folder: 解压后的文件夹名称（可选）
    :return: 解压后的目录
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """
    下载DATA_HUB中的所有文件。
    """
    for name in DATA_HUB:
        download(name)


# 定义Kaggle房价数据集的下载信息
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)

# 损失函数：MSELoss
loss = nn.MSELoss()


def get_net(in_features):
    """
    获取线性网络模型。
    :param in_features: 输入特征数
    :return: 网络模型
    """
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    """
    计算对数根均方误差（log RMSE），用于评估房价预测的相对误差。
    :param net: 网络模型
    :param features: 输入特征
    :param labels: 真实标签
    :return: log RMSE 值
    """
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """
    训练模型。
    :param net: 网络模型
    :param train_features: 训练特征
    :param train_labels: 训练标签
    :param test_features: 测试特征（可选）
    :param test_labels: 测试标签（可选）
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param weight_decay: 权重衰减
    :param batch_size: 批量大小
    :return: 训练损失列表，测试损失列表（如果提供测试数据）
    """
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """
    获取K折交叉验证的第i折数据。
    :param k: 折数
    :param i: 当前折索引
    :param X: 特征数据
    :param y: 标签数据
    :return: 训练特征、训练标签、验证特征、验证标签
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """
    执行K折交叉验证。
    :param k: 折数
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param weight_decay: 权重衰减
    :param batch_size: 批量大小
    :return: 平均训练log RMSE, 平均验证log RMSE
    """
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])  # 注意：这里传入in_features
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            # 可视化第一折的训练和验证损失
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            plt.show()  # 在PyCharm中显示图表
        print(f'折{i + 1}，训练log rmse {float(train_ls[-1]):f}, 验证log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    """
    在全数据集上训练模型，并生成对测试集的预测，保存为CSV文件。
    :param train_features: 训练特征
    :param test_features: 测试特征
    :param train_labels: 训练标签
    :param test_data: 测试数据（pandas DataFrame）
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :param weight_decay: 权重衰减
    :param batch_size: 批量大小
    """
    net = get_net(train_features.shape[1])  # 注意：这里传入in_features
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    # 可视化训练损失
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    plt.show()  # 在PyCharm中显示图表
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
    print("预测结果已保存到 submission.csv")


def main():
    """
    主函数：执行数据下载、预处理、K折交叉验证和最终预测。
    """
    # 下载并读取数据集
    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))

    print(train_data.shape)  # (1460, 81)
    print(test_data.shape)  # (1459, 80)

    # 查看前4行数据的部分特征和标签
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    # 合并训练和测试特征（排除ID和标签）
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 数据预处理：标准化数值特征，填充缺失值为0
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 处理离散特征：独热编码
    all_features = pd.get_dummies(all_features, dummy_na=True)

    # 重要：将所有特征转换为float32类型，以避免numpy object dtype问题
    all_features = all_features.astype('float32')

    print(all_features.shape)  # (2919, 331)

    # 转换为PyTorch张量
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    # 超参数
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

    # 执行K折交叉验证
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')

    # 在全数据集上训练并预测
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)


if __name__ == "__main__":
    main()