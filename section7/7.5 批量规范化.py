import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from d2l import torch as d2l

def main():
    # 设备自动选择
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # 超参数
    lr, num_epochs, batch_size = 1.0, 10, 256

    # 数据加载和预处理
    transform = transforms.ToTensor()
    mnist_train = datasets.FashionMNIST(root='../data', train=True, transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(root='../data', train=False, transform=transform, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # 模型结构
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    net = net.to(device)

    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    d2l.plt.show()

if __name__ == "__main__":
    main()
