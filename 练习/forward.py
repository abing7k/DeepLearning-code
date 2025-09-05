import torch
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        print("输入 X 的形状:", X.shape)  # (batch_size, num_steps)

        # 1. 词嵌入
        X = self.embedding(X)
        print("经过 embedding:", X.shape)  # (batch_size, num_steps, embed_size)

        # 2. 维度转换
        X = X.permute(1, 0, 2)
        print("permute 后:", X.shape)  # (num_steps, batch_size, embed_size)

        # 3. RNN 前向传播
        output, state = self.rnn(X)
        print("RNN output:", output.shape)  # (num_steps, batch_size, num_hiddens)
        print("RNN state :", state.shape)   # (num_layers, batch_size, num_hiddens)

        return output, state


def main():
    # 假设：词表大小=20，embedding维度=8，隐藏层=16，层数=2
    vocab_size, embed_size, num_hiddens, num_layers = 20, 8, 16, 2
    batch_size, num_steps = 4, 5  # batch=4，句子长度=5

    # 模型
    encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)

    # 随机输入 (batch_size, num_steps)，里面是词的索引
    X = torch.randint(0, vocab_size, (batch_size, num_steps))

    # 前向传播
    output, state = encoder(X)


if __name__ == "__main__":
    main()
