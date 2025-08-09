import torch
from torch import nn
from d2l import torch as d2l


# 构建 LeNet 网络
def build_lenet():
    net = nn.Sequential(
        # 第1个卷积层：输入通道1，输出通道6，卷积核5x5，padding=2 保持尺寸
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        # 平均池化层：窗口 2x2，步幅 2（高宽减半）
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 第2个卷积层：输入通道6，输出通道16，卷积核5x5（无 padding，高宽减少4）
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        # 平均池化层
        nn.AvgPool2d(kernel_size=2, stride=2),

        # 展平成一维向量
        nn.Flatten(),

        # 全连接层1
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        # 全连接层2
        nn.Linear(120, 84), nn.Sigmoid(),
        # 输出层
        nn.Linear(84, 10)
    )
    return net


# GPU/MPS 精度评估
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用 GPU 或 MPS 计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用 GPU 或 MPS 训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel='epoch', xlim=[1, num_epochs],
        legend=['train loss', 'train acc', 'test acc']
    )
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def main():
    # 构建模型
    net = build_lenet()

    # 加载 Fashion-MNIST 数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

    # 训练参数
    lr, num_epochs = 0.9, 10

    # 自动检测设备（优先 MPS，再 CUDA，最后 CPU）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 训练计时
    start_time = d2l.Timer()
    # 训练
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    print(f"Total training time: {start_time.stop():.2f} sec")
    # 确保在 PyCharm 中显示 epoch 曲线
    d2l.plt.show()


if __name__ == "__main__":
    main()