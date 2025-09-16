import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt  # 如果你后续需要绘图
from d2l import torch as d2l  # 确保你安装了 d2l 库：pip install d2l
import random

def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声"""
    # 从正态分布生成特征矩阵 X，形状为 (num_examples, len(w))
    X = torch.normal(0, 1, (num_examples, len(w)))
    print(X)
    # 计算真实的标签 y = Xw + b
    y = torch.matmul(X, w) + b
    # 添加高斯噪声，使得 y 不那么完美
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征和标签
    return X, y.reshape((-1, 1))

# 设定真实参数 w 和 b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成1000个样本
features, labels = synthetic_data(true_w, true_b, 1000)

# 打印第一个样本的特征和对应的标签
print('features:', features[0])
print('label:', labels[0])

# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
# d2l.plt.scatter(features[:, (0)].detach().numpy(), labels.detach().numpy(), 1);
# plt.show()


# 3.2.2


# 定义一个函数：用于生成小批量数据
def data_iter(batch_size, features, labels):
    """
    输入：
        batch_size：每个批次的样本数量（如10）
        features：特征张量，形状为(num_examples, num_features)
        labels：标签张量，形状为(num_examples, 1)
    输出：
        每次返回一小批特征和对应的标签
    """

    # 样本总数，例如 1000 个样本
    num_examples = len(features)

    # 生成从 0 到 num_examples-1 的索引列表，例如 [0, 1, 2, ..., 999]
    indices = list(range(num_examples))

    # 将索引随机打乱（打乱顺序是为了打破数据间的顺序相关性，有利于训练）
    random.shuffle(indices)

    # 从0开始，每次取 batch_size 个索引，直到遍历完整个数据集
    for i in range(0, num_examples, batch_size):
        # 取出当前批次的索引（注意末尾可能不足一个 batch）
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )

        # 使用这些索引从 features 和 labels 中选出当前批次的数据
        yield features[batch_indices], labels[batch_indices]


# ======== 以下是生成模拟数据的代码，用于测试 data_iter 函数 ==========

# 设置样本数量
num_examples = 1000

# 设置真实权重和偏置，用于生成标签（模拟一个线性模型）
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成特征 X，维度为 (1000, 2)，其中每个元素来自标准正态分布
features = torch.normal(0, 1, (num_examples, len(true_w)))

# 计算标签 y = Xw + b，并加入一点高斯噪声
labels = torch.matmul(features, true_w) + true_b
labels += torch.normal(0, 0.01, labels.shape)

# 把标签 reshaped 成二维张量 (1000, 1)，便于后续模型计算
labels = labels.reshape((-1, 1))

# ======== 使用 data_iter 函数读取一个小批量样本并打印 ==========

# 设置批量大小
batch_size = 10

# 使用 for 循环从数据集中获取一个 batch（只取第一个，后面就 break 了）
for X, y in data_iter(batch_size, features, labels):
    print("小批量特征 X:\n", X)
    print("\n对应标签 y:\n", y)
    break  # 只演示一个小批量





import torch
from torch.utils.data import TensorDataset, DataLoader
import random

# 1. 生成数据集（y = Xw + b + 噪声）
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 特征 X
    y = torch.matmul(X, w) + b                      # 线性组合
    y += torch.normal(0, 0.01, y.shape)             # 加噪声
    return X, y.reshape((-1, 1))                    # 返回X和列向量y

# 真实参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 2. 数据加载器（小批量）
def load_array(data_arrays, batch_size, is_train=True):
    """构造数据加载器"""
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 3. 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
print("w初始：\n",w)
b = torch.zeros(1, requires_grad=True)

# 4. 定义线性回归模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 5. 均方损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 6. 随机梯度下降优化器
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 更新
            param.grad.zero_()                     # 梯度清零

# 7. 模型训练
lr = 0.03          # 学习率
num_epochs = 3     # 训练轮数
net = linreg       # 模型
loss = squared_loss  # 损失函数

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X, w, b), y)      # 小批量损失
        l.sum().backward()             # 反向传播计算梯度
        sgd([w, b], lr, batch_size)    # 使用SGD更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 8. 评估参数误差
print("true_w\n",true_w)
print("w\n",w)
print("w.reshape(true_w.shape)\n",w.reshape(true_w.shape))
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')

print(f'b的估计误差: {true_b - b}')
