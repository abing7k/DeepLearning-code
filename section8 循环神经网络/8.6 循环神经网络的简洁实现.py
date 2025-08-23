import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def try_gpu():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# 8.6 循环神经网络的简洁实现
def main():
    # =====================
    # 8.6.1 定义模型
    # =====================
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    num_hiddens = 256
    # 构造一个单隐藏层的 RNN
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # 初始化隐状态，形状: (隐藏层数, 批量大小, 隐藏单元数)
    state = torch.zeros((1, batch_size, num_hiddens))
    print(f"state.shape: {state.shape}")

    # 测试单步计算
    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(f"Y.shape: {Y.shape}, state_new.shape: {state_new.shape}")

    # =====================
    # 定义完整的 RNN 模型
    # =====================
    class RNNModel(nn.Module):
        """循环神经网络模型"""
        def __init__(self, rnn_layer, vocab_size, **kwargs):
            super(RNNModel, self).__init__(**kwargs)
            self.rnn = rnn_layer
            self.vocab_size = vocab_size
            self.num_hiddens = self.rnn.hidden_size
            # 判断是否双向
            self.num_directions = 2 if self.rnn.bidirectional else 1
            # 输出层
            self.linear = nn.Linear(self.num_hiddens * self.num_directions,
                                    self.vocab_size)

        def forward(self, inputs, state):
            # inputs: (批量大小, 时间步数)
            X = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32)
            Y, state = self.rnn(X, state)
            # Y: (时间步数, 批量大小, 隐藏单元数)
            # reshape -> (时间步数*批量大小, 隐藏单元数)
            output = self.linear(Y.reshape((-1, Y.shape[-1])))
            return output, state

        def begin_state(self, device, batch_size=1):
            """初始化隐状态"""
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU/nn.RNN 隐状态是张量
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                                    batch_size, self.num_hiddens),
                                   device=device)
            else:
                # nn.LSTM 隐状态是元组 (h, c)
                return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                     batch_size, self.num_hiddens), device=device),
                        torch.zeros((self.num_directions * self.rnn.num_layers,
                                     batch_size, self.num_hiddens), device=device))

    # =====================
    # 8.6.2 训练与预测
    # =====================
    device = try_gpu()
    print(f"Using device: {device}")
    net = RNNModel(rnn_layer, vocab_size=len(vocab)).to(device)

    # 使用随机权重的预测（效果很差）
    print(d2l.predict_ch8('time traveller', 10, net, vocab, device))

    # 训练模型
    num_epochs, lr = 500, 1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
plt.show()