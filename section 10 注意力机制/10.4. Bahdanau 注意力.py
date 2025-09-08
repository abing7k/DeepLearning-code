import time
import torch
from torch import nn
from d2l import torch as d2l


# ========================
# 带注意力机制的解码器基类
# ========================
# @save
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


# ========================
# Bahdanau 注意力解码器
# ========================
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 加性注意力
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # RNN 解码器
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        # 输出层
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """
        初始化解码器状态
        enc_outputs: (outputs, hidden_state)
            - outputs: 编码器在所有时间步的隐状态 (batch_size, num_steps, num_hiddens)
            - hidden_state: 编码器的最后一层隐状态 (num_layers, batch_size, num_hiddens)
        """
        outputs, hidden_state = enc_outputs
        # 转换维度，方便做注意力 (num_steps, batch_size, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        """
        解码器前向传播
        X: (batch_size, num_steps) 解码器输入
        state: 编码器输出的状态
        """
        enc_outputs, hidden_state, enc_valid_lens = state
        # 词嵌入并调整维度 => (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []

        for x in X:  # 遍历每个时间步
            # query = 上一个时间步的最后一层隐状态
            query = torch.unsqueeze(hidden_state[-1], dim=1)  # (batch_size, 1, num_hiddens)
            # context = 注意力上下文向量
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 拼接输入
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # RNN 解码
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        # 拼接输出并映射到词表大小
        outputs = self.dense(torch.cat(outputs, dim=0))  # (num_steps, batch_size, vocab_size)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


# ========================
# 主函数
# ========================
def main():
    # 设备检测：优先cuda，其次mps，最后cpu
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    # ==============
    # 10.4.2 测试解码器
    # ==============
    encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8,
                                 num_hiddens=16, num_layers=2)
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8,
                                      num_hiddens=16, num_layers=2)
    encoder.eval(), decoder.eval()

    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size, num_steps)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
    print("10.4.2 测试输出:")
    print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)

    # ==============
    # 10.4.3 训练
    # ==============
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 250

    # 加载数据
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    # 定义模型
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)

    # 训练
    net = net.to(device)
    start = time.time()
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    print(f"训练完成，耗时 {time.time() - start:.2f} 秒")

    # ==============
    # 翻译测试 & BLEU
    # ==============
    engs = ['go .', "i lost .", "he's calm .", "i'm home ."]
    fras = ['va !', "j'ai perdu .", "il est calme .", "je suis chez moi ."]

    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

    # ==============
    # 注意力权重可视化
    # ==============
    attention_weights = torch.cat(
        [step[0][0][0] for step in dec_attention_weight_seq], 0
    ).reshape((1, 1, -1, num_steps))

    d2l.show_heatmaps(
        attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
        xlabel='Key positions', ylabel='Query positions')
    d2l.plt.show()


if __name__ == "__main__":
    main()
