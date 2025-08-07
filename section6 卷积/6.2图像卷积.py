import torch

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def main():
    # 输入张量 X 和 卷积核 K
    X = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0]
    ])
    K = torch.tensor([
        [0.0, 1.0],
        [2.0, 3.0]
    ])

    # 计算二维互相关
    Y = corr2d(X, K)
    print("输入 X:\n", X)
    print("卷积核 K:\n", K)
    print("输出 Y = corr2d(X, K):\n", Y)


# 自定义二维卷积层
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def edge_detection_demo():
    # 创建黑白边图像
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print("输入图像 X:\n", X)

    # 创建用于检测垂直边缘的卷积核
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print("垂直边缘检测输出 Y:\n", Y)

    # 对图像转置进行同样的操作（检测水平边缘）
    Y_t = corr2d(X.t(), K)
    print("转置图像的边缘检测输出（应该检测不到边缘）:\n", Y_t)

def learn_kernel_demo():
    print("\n=== 学习卷积核示例 ===")
    # 原始输入图像
    X = torch.ones((6, 8))
    X[:, 2:6] = 0

    # 目标输出（边缘）
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)

    # 构造一个二维卷积层，它具有1个输出通道和形状为(1, 2)的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    # 该二维卷积层使用四维输入输出格式：(batch_size, channels, height, width)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    lr = 3e-2  # 学习率

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')

    print("学习到的卷积核权重:\n", conv2d.weight.data)

if __name__ == "__main__":
    main()
    print("\n=== 边缘检测示例 ===")
    edge_detection_demo()
    learn_kernel_demo()