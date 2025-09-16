import math
import time
import torch
from torch import nn
from d2l import torch as d2l

# 10.6.1 自注意力演示
# 使用 d2l.MultiHeadAttention 在张量上计算自注意力

def test_self_attention():
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(
        num_hiddens, num_hiddens, num_hiddens,
        num_hiddens, num_heads, 0.5
    )
    attention.eval()  # 不启用 dropout
    print(attention)

    batch_size, num_queries = 2, 4
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = attention(X, X, X, valid_lens)
    print("自注意力输出形状:", Y.shape)


# 10.6.2 卷积、循环神经网络和自注意力复杂度比较
# 这里只是理论部分，没有代码实现

# 10.6.3 位置编码实现
class PositionalEncoding(nn.Module):
    """位置编码 (基于正弦和余弦函数)"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的位置编码矩阵 P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # 将位置编码加到输入表示上
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


def test_positional_encoding():
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]

    # 绘制第6到第9列的位置编码
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T,
             xlabel='Row (position)', figsize=(6, 2.5),
             legend=["Col %d" % d for d in torch.arange(6, 10)])
    d2l.plt.show()

    # 打印0到7的二进制形式
    for i in range(8):
        print(f'{i}的二进制是：{i:>03b}')

    # 展示热图
    P_show = P[0, :, :].unsqueeze(0).unsqueeze(0)
    d2l.show_heatmaps(P_show, xlabel='Column (encoding dimension)',
                      ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
    d2l.plt.show()


def main():
    start_time = time.time()

    print("===== 测试自注意力 =====")
    test_self_attention()

    print("===== 测试位置编码 =====")
    test_positional_encoding()

    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()