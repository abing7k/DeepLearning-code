# ==============================
# 10.5.2 多头注意力实现
# ==============================
import torch
from torch import nn
from d2l import torch as d2l


# ==============================
# 多头注意力机制
# ==============================
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        # 缩放点积注意力
        self.attention = d2l.DotProductAttention(dropout)
        # 分别为 Q, K, V 定义线性变换
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        # 输出层
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        queries, keys, values: (batch_size, 序列长度, num_hiddens)
        valid_lens: (batch_size,) 或 (batch_size, 序列长度)
        """
        # 线性变换 + 分头
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 把 valid_lens 在 batch 维度复制 num_heads 次
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 注意力计算
        output = self.attention(queries, keys, values, valid_lens)

        # 把多头拼接回来
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# ==============================
# 工具函数
# ==============================
def transpose_qkv(X, num_heads):
    """
    为了多头注意力的并行计算而变换形状
    输入: (batch_size, 序列长度, num_hiddens)
    输出: (batch_size*num_heads, 序列长度, num_hiddens/num_heads)
    """
    # 先 reshape: (batch_size, 序列长度, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 调整维度顺序: (batch_size, num_heads, 序列长度, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 合并 batch_size 和 num_heads 维度
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    逆转 transpose_qkv 的操作
    输入: (batch_size*num_heads, 序列长度, num_hiddens/num_heads)
    输出: (batch_size, 序列长度, num_hiddens)
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# ==============================
# main 函数 (测试代码)
# ==============================
def main():
    # 设置参数
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(
        key_size=num_hiddens,
        query_size=num_hiddens,
        value_size=num_hiddens,
        num_hiddens=num_hiddens,
        num_heads=num_heads,
        dropout=0.5
    )
    attention.eval()  # 评估模式 (关闭 dropout)

    # 打印模型结构
    print(attention)

    # 构造输入 (书中例子)
    batch_size, num_queries, num_kvpairs = 2, 4, 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))

    # 运行多头注意力
    out = attention(X, Y, Y, valid_lens)

    # 打印输出形状 (书中结果: torch.Size([2, 4, 100]))
    print("输出形状:", out.shape)

    # 画一个示意图 (这里只画随机 attention 权重的热力图)
    d2l.show_heatmaps(
        torch.rand((1, num_heads, num_queries, num_kvpairs)),
        xlabel='Keys', ylabel='Queries',
        titles=[f'Head {i}' for i in range(1, num_heads + 1)],
        figsize=(7, 3)
    )
    d2l.plt.show()


if __name__ == "__main__":
    main()
