import torch
from torch import nn
from d2l import torch as d2l


def load_data(batch_size=256):
    """加载 Fashion-MNIST 数据集"""
    return d2l.load_data_fashion_mnist(batch_size)


def create_net():
    """创建 Softmax 回归模型"""
    net = nn.Sequential(
        nn.Flatten(),           # 将 [batch, 1, 28, 28] 展平为 [batch, 784]
        nn.Linear(784, 10)      # 输出10类
    )
    return net


def init_weights(m):
    """初始化权重为 N(0, 0.01)"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def train(net, train_iter, test_iter, loss, num_epochs, trainer):
    """训练模型，使用 D2L 封装函数"""
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


d2l.train

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（用于softmax简洁实现）"""
    d2l.use_svg_display()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)  # train_loss_sum, train_acc_sum, num_examples
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
            with torch.no_grad():
                metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')


def main():
    # 超参数
    batch_size = 256
    lr = 0.1
    num_epochs = 10

    # 1. 加载数据
    train_iter, test_iter = load_data(batch_size)

    # 2. 创建模型并初始化参数
    net = create_net()
    net.apply(init_weights)

    # 3. 定义损失函数（内置数值稳定的 softmax + log + NLLLoss）
    loss = nn.CrossEntropyLoss(reduction='none')

    # 4. 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 5. 开始训练
    train(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    main()
