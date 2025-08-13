import time
import torch
from torch import nn
from d2l import torch as d2l


# 定义一个卷积块（BatchNorm -> ReLU -> Conv）
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# 定义稠密块（Dense Block）
# 每个卷积块的输入 = 之前所有层的输出拼接
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 在通道维度拼接输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


# 定义过渡层（Transition Layer）
# 1×1 卷积降低通道数 + 平均池化降低高和宽
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def main():
    # -----------------------
    # 1. 设备选择（优先CUDA，其次MPS，最后CPU）
    # -----------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # -----------------------
    # 2. DenseNet 模型结构
    # -----------------------

    # 第一部分：初始卷积层（与ResNet类似）
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # 定义稠密块与过渡层
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        num_channels += num_convs * growth_rate
        # 除最后一个稠密块外，后面都接过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    # 最终结构：稠密块 + BN + ReLU + 全局平均池化 + 全连接层
    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    )

    # -----------------------
    # 3. 数据加载（Fashion-MNIST，96×96）
    # -----------------------
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    # -----------------------
    # 4. 训练模型（计时）
    # -----------------------
    start_time = time.time()
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    end_time = time.time()

    print(f"Training finished in {end_time - start_time:.2f} seconds")
    d2l.plt.show()

if __name__ == "__main__":
    main()