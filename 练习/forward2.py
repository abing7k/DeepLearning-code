import torch
from torch import nn


# ================= 10.7.2 基于位置的前馈网络 =================
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        print("输入 X：", X.shape)
        print(X)  # 打印原始输入

        H = self.dense1(X)
        print("\n经过 dense1 后：", H.shape)
        print(H)

        H = self.relu(H)
        print("\n经过 ReLU 后：", H.shape)
        print(H)

        Y = self.dense2(H)
        print("\n经过 dense2 后（最终输出）：", Y.shape)
        print(Y)
        return Y


# ================= 使用例子 =================
if __name__ == "__main__":
    # 输入：batch_size=2, 序列长度=3, 每个 token 是4维
    X = torch.ones((2, 3, 4))
    ffn = PositionWiseFFN(4, 8, 4)  # 输入4维 → 隐藏8维 → 输出4维

    Y = ffn(X)
    print("\n最终 Y：", Y.shape)
