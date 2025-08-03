import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


# 手动实现 dropout 层
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 生成与X形状相同的mask，从[0,1)中采样
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)  # 进行缩放以保持期望值不变


# 定义包含两个隐藏层的MLP，并应用dropout
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = 0.2  # 第一层的dropout概率
        self.dropout2 = 0.5  # 第二层的dropout概率

    def forward(self, X):
        H1 = self.relu(self.lin1(X))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        out = self.lin3(H2)
        return out


# 计算精度
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat == y).sum())


# 每轮训练
def train_epoch_ch3(net, train_iter, loss, updater):
    net.train()
    metric = [0.0, 0.0]  # [累积损失, 正确预测数]
    for X, y in train_iter:
        X, y = X.view(X.shape[0], -1), y
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.backward()
        updater.step()
        metric[0] += float(l.sum())
        metric[1] += accuracy(y_hat, y)
    return metric[0] / len(train_iter.dataset), metric[1] / len(train_iter.dataset)


# 测试准确率
def evaluate_accuracy(net, data_iter):
    net.eval()
    correct = 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.view(X.shape[0], -1)
            correct += accuracy(net(X), y)
    return correct / len(data_iter.dataset)


# 完整训练过程
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    train_acc_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'epoch {epoch + 1}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    # 可视化准确率变化
    plt.plot(range(1, num_epochs + 1), train_acc_list, label='train acc')
    plt.plot(range(1, num_epochs + 1), test_acc_list, label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主函数
def main():
    batch_size = 256
    # 加载 Fashion-MNIST 数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    # 初始化网络
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    # 损失函数和优化器
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=0.5)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


if __name__ == '__main__':
    main()
