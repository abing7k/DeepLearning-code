import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # 对输入 X 和卷积核 K 在通道维度 zip 配对，逐通道执行 d2l.corr2d 并求和
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # 对 K 的第 0 维迭代，调用 corr2d_multi_in，最后用 torch.stack 堆叠结果
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


def corr2d_multi_in_out_1x1(X, K):
    # 实现 1x1 卷积，将输入 X reshape 成 (c_i, h*w)，K reshape 成 (c_o, c_i)，用 torch.matmul 计算，最后 reshape 回 (c_o, h, w)
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape(c_i, h * w)
    K = K.reshape(c_o, c_i)
    Y = torch.matmul(K, X)
    return Y.reshape(c_o, h, w)


def main():
    # 创建示例输入 X（2 通道）和核 K（2 通道），验证 corr2d_multi_in 输出
    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = torch.tensor([[[0, 1], [2, 3]],
                      [[1, 2], [3, 4]]])
    print("corr2d_multi_in(X, K) =\n", corr2d_multi_in(X, K))

    # 构造多输出通道的卷积核 K（原核、K+1、K+2 堆叠），验证 corr2d_multi_in_out 输出
    K_multi_out = torch.stack([K, K + 1, K + 2], 0)
    print("corr2d_multi_in_out(X, K_multi_out) =\n", corr2d_multi_in_out(X, K_multi_out))

    # 用随机输入 (3, 3, 3) 和核 (2, 3, 1, 1) 验证 1x1 卷积与 corr2d_multi_in_out 输出一致
    X_random = torch.randn(3, 3, 3)
    K_1x1 = torch.randn(2, 3, 1, 1)
    # 直接将 K_1x1 传递给 corr2d_multi_in_out
    Y1 = corr2d_multi_in_out(X_random, K_1x1)
    Y2 = corr2d_multi_in_out_1x1(X_random, K_1x1)
    print("corr2d_multi_in_out with random input =\n", Y1)
    print("corr2d_multi_in_out_1x1 with random input =\n", Y2)
    print("Difference =\n", (Y1 - Y2).abs().sum())
    print("Difference =\n", (Y1 - Y2).abs().sum())


# ====== 6.4.3 1x1 卷积验证 ======
X_demo = torch.normal(0, 1, (3, 3, 3))
K_demo = torch.normal(0, 1, (2, 3, 1, 1))
Y1_demo = corr2d_multi_in_out_1x1(X_demo, K_demo)
Y2_demo = corr2d_multi_in_out(X_demo, K_demo)
print("1x1 conv result:\n", Y1_demo)
print("Multi-in-out conv result:\n", Y2_demo)
print("Difference for 1x1 conv =\n", (Y1_demo - Y2_demo).abs().sum())
assert float(torch.abs(Y1_demo - Y2_demo).sum()) < 1e-6


if __name__ == "__main__":
    main()
