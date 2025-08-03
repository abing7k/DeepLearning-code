import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# 设置设备为 CPU（默认）
device = torch.device("cpu")

# 加载 Fashion-MNIST 数据
def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    mnist_train = FashionMNIST(root="../data", train=True, transform=transform, download=True)
    mnist_test = FashionMNIST(root="../data", train=False, transform=transform, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter

# 初始化权重：使用正态分布初始化全连接层权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

# 计算准确率
def evaluate_accuracy(net, data_iter):
    net.eval()  # 设置评估模式，关闭dropout
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

# 训练函数（手动实现 train_ch3）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    train_loss_all, train_acc_all, test_acc_all = [], [], []

    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.backward()
            updater.step()

            total_loss += l.item() * y.shape[0]
            correct += (y_hat.argmax(1) == y).sum().item()
            total += y.numel()

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate_accuracy(net, test_iter)

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)

        print(f'epoch {epoch+1}, loss {train_loss:.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    return train_loss_all, train_acc_all, test_acc_all

# 可视化训练过程
def plot_metrics(epochs, train_loss, train_acc, test_acc):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'bo-', label='Train Acc')
    plt.plot(epochs, test_acc, 'r^-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'gs-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------
# 主程序入口 main()
# ---------------------
def main():
    # 超参数设置
    batch_size = 256
    lr = 0.5
    num_epochs = 10
    dropout1 = 0.2
    dropout2 = 0.5

    # 加载数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 构建模型
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10)
    )

    net.apply(init_weights)

    # 损失函数与优化器
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练与测试
    start = time.time()
    train_loss, train_acc, test_acc = train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print(f"训练完成，总耗时：{time.time() - start:.2f} 秒")

    # 可视化
    epochs = list(range(1, num_epochs + 1))
    plot_metrics(epochs, train_loss, train_acc, test_acc)

# 运行 main 函数
if __name__ == '__main__':
    main()
