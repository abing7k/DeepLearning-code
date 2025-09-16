import torch
from torch import nn
from torch.nn import functional as F


# 定义残差块
class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def main():
    # 测试 1：输入输出形状一致
    blk1 = Residual(3, 3)
    X1 = torch.rand(4, 3, 6, 6)
    Y1 = blk1(X1)
    print("Test 1 - Same channels, same size")
    print("Input shape:", X1.shape)
    print("Output shape:", Y1.shape)

    # 测试 2：增加通道数并减半高宽
    blk2 = Residual(3, 6, use_1x1conv=True, strides=2)
    X2 = torch.rand(4, 3, 6, 6)
    Y2 = blk2(X2)
    print("\nTest 2 - Increase channels, halve size")
    print("Input shape:", X2.shape)
    print("Output shape:", Y2.shape)


if __name__ == "__main__":
    main()