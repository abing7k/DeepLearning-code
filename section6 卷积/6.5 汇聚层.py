import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

def main():
    # 创建一个3x3的张量X，作为输入数据
    X = torch.tensor([[0.0, 1.0, 2.0],
                      [3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0]])
    # 使用自定义的pool2d函数进行最大池化，池化窗口大小为2x2
    print("自定义最大池化结果:")
    print(pool2d(X, (2, 2), 'max'))

    # 使用自定义的pool2d函数进行平均池化，池化窗口大小为2x2
    print("自定义平均池化结果:")
    print(pool2d(X, (2, 2), 'avg'))

    # 创建一个4D张量X，形状为(1, 1, 4, 4)，用于后续的PyTorch池化示例
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)

    # 使用PyTorch内置的MaxPool2d进行池化，窗口大小为3x3，无填充，步长默认
    m = nn.MaxPool2d(3)
    print("PyTorch MaxPool2d(3)池化结果:")
    print(m(X))

    # 使用带填充(padding=1)和步长(stride=2)的MaxPool2d进行池化
    m = nn.MaxPool2d(3, padding=1, stride=2)
    print("PyTorch MaxPool2d(3, padding=1, stride=2)池化结果:")
    print(m(X))

    # 将X沿第1维拼接，构造多通道输入
    X_multi = torch.cat((X, X + 1), 1)
    print("多通道输入张量形状:")
    print(X_multi)
    print(X_multi.shape)

    # 对多通道输入使用带填充和步长的MaxPool2d进行池化
    m = nn.MaxPool2d(3, padding=1, stride=2)
    print("多通道输入的池化结果:")
    print(m(X_multi))

if __name__ == '__main__':
    main()
