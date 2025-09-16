import time  # 用于记录训练时间
import torch
from torch import nn
import torchvision
from d2l import torch as d2l

# Inception 块定义，含并行结构
class Inception(nn.Module):
    # c1 - c4为每条路径输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 第一条路径：1x1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 第二条路径：1x1卷积后接3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 第三条路径：1x1卷积后接5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 第四条路径：3x3最大池化后接1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    def forward(self, x):
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        # 并行路径输出在通道维concat
        return torch.cat((p1, p2, p3, p4), dim=1)


def main():
    # 构建GoogLeNet网络结构，分为若干块
    # 第一块：单层卷积+池化
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第二块：连续卷积+池化
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第三块：2个Inception块+池化
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第四块：5个Inception块+池化
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # 第五块：2个Inception块+全局平均池化
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten())
    # 拼接所有模块
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.Linear(1024, 10))

    # 输出各模块输出形状，便于调试
    X = torch.rand(1, 1, 96, 96)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 训练参数
    batch_size = 128
    # 下载Fashion-MNIST数据集，并调整图片大小
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    # 训练GoogLeNet
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    lr, num_epochs = 0.1, 10
    print('Starting training...')
    net.to(device)
    # 记录训练开始时间
    start_time = time.time()
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    d2l.plt.show()
    # 记录训练结束时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')


# 允许直接运行本文件
if __name__ == "__main__":
    main()