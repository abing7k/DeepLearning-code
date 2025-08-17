# -*- coding: utf-8 -*-
"""
序列模型预测示例（PyTorch + d2l）
包含：
1. 原始时间序列
2. 单步预测
3. 多步滚动预测
4. 不同步长的多步预测对比
"""

import torch
from torch import nn
from d2l import torch as d2l

# =========================
# 初始化网络权重
# =========================
def init_weights(m):
    """Xavier 均匀分布初始化全连接层权重"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# =========================
# 构建多层感知机模型
# =========================
def get_net():
    """
    输入：长度为 tau 的时间窗口
    输出：预测下一个时间点的值
    """
    net = nn.Sequential(
        nn.Linear(4, 10),   # 输入4个特征 → 隐藏层10个神经元
        nn.ReLU(),
        nn.Linear(10, 1)    # 隐藏层10个神经元 → 输出1个预测值
    )
    net.apply(init_weights)
    return net

# =========================
# 训练函数
# =========================
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

# =========================
# 主程序
# =========================
def main():
    # ---------- 1. 生成数据 ----------
    T = 1000  # 总共产生1000个点
    time = torch.arange(1, T + 1, dtype=torch.float32)
    # 生成带噪声的正弦波：sin(0.01 * t) + N(0, 0.2)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    tau = 4  # 时间窗口长度

    # 画图1：原始数据
    d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    # ---------- 2. 构造特征和标签 ----------
    # features.shape = (T - tau, tau)
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]  # 每列是一个滞后特征
    labels = x[tau:].reshape((-1, 1))       # 标签是下一个时刻的值

    batch_size, n_train = 16, 600
    # 前 n_train 条样本用于训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                                batch_size, is_train=True)

    # ---------- 3. 训练模型 ----------
    net = get_net()
    loss = nn.MSELoss(reduction='none')
    train(net, train_iter, loss, epochs=5, lr=0.01)

    # ---------- 4. 单步预测 ----------
    onestep_preds = net(features)
    # 画图2：单步预测 vs 原始数据
    d2l.plot([time, time[tau:]],
             [x.detach().numpy(), onestep_preds.detach().numpy()],
             'time', 'x', legend=['data', '1-step preds'],
             xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    # ---------- 5. 多步滚动预测 ----------
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train + tau] = x[: n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(
            multistep_preds[i - tau:i].reshape((1, -1))
        )
    # 画图3：单步预测 vs 多步滚动预测 vs 原始数据
    d2l.plot([time, time[tau:], time[n_train + tau:]],
             [x.detach().numpy(),
              onestep_preds.detach().numpy(),
              multistep_preds[n_train + tau:].detach().numpy()],
             'time', 'x',
             legend=['data', '1-step preds', 'multistep preds'],
             xlim=[1, 1000], figsize=(6, 3))
    d2l.plt.show()

    # ---------- 6. 不同步长的多步预测 ----------
    max_steps = 64
    # 特征矩阵：前 tau 列是真实历史，后面列是滚动预测
    features_multi = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    # 填充前 tau 列真实值
    for i in range(tau):
        features_multi[:, i] = x[i: i + T - tau - max_steps + 1]
    # 填充滚动预测列
    for i in range(tau, tau + max_steps):
        features_multi[:, i] = net(features_multi[:, i - tau:i]).reshape(-1)

    steps = (1, 4, 16, 64)
    # 画图4：不同预测步长曲线
    d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
             [features_multi[:, (tau + i - 1)].detach().numpy() for i in steps],
             'time', 'x',
             legend=[f'{i}-step preds' for i in steps],
             xlim=[5, 1000], figsize=(6, 3))
    d2l.plt.show()

if __name__ == '__main__':
    main()