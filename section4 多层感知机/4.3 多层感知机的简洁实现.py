import torch
from torch import nn
from d2l import torch as d2l
from section3 import softmax_scratch36

# ---------------------------
# 定义模型结构
# ---------------------------
def build_mlp():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),  # 输入层784 -> 隐藏层256
        nn.ReLU(),            # ReLU 激活函数
        nn.Linear(256, 10)    # 隐藏层 -> 输出层10
    )
    return net

# ---------------------------
# 权重初始化函数
# ---------------------------
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# ---------------------------
# 主函数入口
# ---------------------------
def main():
    # 模型构建与初始化
    net = build_mlp()
    net.apply(init_weights)

    # 超参数设置
    batch_size, lr, num_epochs = 256, 0.1, 10

    # 损失函数与优化器
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 加载Fashion-MNIST数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 模型训练
    softmax_scratch36.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# ---------------------------
# 程序入口
# ---------------------------
if __name__ == '__main__':
    main()