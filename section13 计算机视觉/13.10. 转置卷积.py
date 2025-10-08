# -*- coding: utf-8 -*-
"""
13.10 转置卷积

重现《动手学深度学习》中对转置卷积的讲解示例：
- 逐步实现基础转置卷积运算
- 演示 PyTorch API 中的填充、步幅和多通道行为
- 展示转置卷积与矩阵乘法的联系

无训练流程，默认使用 CPU。
"""
import torch
from torch import nn
from d2l import torch as d2l


# ---------------------------
# 13.10.1 基本操作
# ---------------------------
def trans_conv(X: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """实现二维基础转置卷积。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y


def demo_basic_transposed_conv():
    """使用书中示例验证基础转置卷积实现。"""
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = trans_conv(X, K)
    print('[基础转置卷积输出]')
    print(Y)

    X4d, K4d = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
    tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
    tconv.weight.data = K4d
    Y_api = tconv(X4d)
    print('[ConvTranspose2d 输出]')
    print(Y_api)


# ---------------------------
# 13.10.2 填充、步幅与多通道
# ---------------------------
def demo_padding_stride():
    """展示填充和步幅在转置卷积中的作用。"""
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 1, 2, 2)
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]]).reshape(1, 1, 2, 2)

    tconv_pad = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
    tconv_pad.weight.data = K
    print('[padding=1 输出]')
    print(tconv_pad(X))

    tconv_stride = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    tconv_stride.weight.data = K
    print('[stride=2 输出]')
    print(tconv_stride(X))


def demo_multichannel_equivalence():
    """验证转置卷积恢复卷积前的形状。"""
    X = torch.rand(size=(1, 10, 16, 16))
    conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
    tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
    result = tconv(conv(X)).shape == X.shape
    print('[多通道形状对齐验证]')
    print(result)


# ---------------------------
# 13.10.3 与矩阵变换的联系
# ---------------------------
def kernel2matrix(K: torch.Tensor) -> torch.Tensor:
    """将 2×2 卷积核扩展为稀疏矩阵表示。"""
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W


def demo_matrix_relationship():
    """演示卷积/转置卷积与矩阵乘法的等价关系。"""
    X = torch.arange(9.0).reshape(3, 3)
    K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    Y = d2l.corr2d(X, K)
    print('[常规卷积输出]')
    print(Y)

    W = kernel2matrix(K)
    matmul_result = torch.matmul(W, X.reshape(-1)).reshape(2, 2)
    print('[矩阵乘法实现卷积]')
    print(matmul_result)

    Z = trans_conv(Y, K)
    matmul_transposed = torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)
    print('[转置卷积输出]')
    print(Z)
    print('[矩阵乘法实现转置卷积]')
    print(matmul_transposed)


# ---------------------------
# main() 入口
# ---------------------------
def main():
    demo_basic_transposed_conv()
    demo_padding_stride()
    demo_multichannel_equivalence()
    demo_matrix_relationship()


if __name__ == '__main__':
    main()
