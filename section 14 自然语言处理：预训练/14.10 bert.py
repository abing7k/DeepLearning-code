import hashlib
import math
import os
import random
import time
import zipfile
import tarfile

import requests
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import d2l.torch as d2l

# ------------------ 设备检测 ------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')

# ------------------ 数据处理相关 ------------------
def read_wiki(data_dir):
    file_path = os.path.join(data_dir, 'wiki.train.tokens')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请确保数据已放置在正确路径。")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 过滤掉空行
    lines = [l.strip() for l in lines if len(l.strip()) > 0 and not l.startswith('=')]
    return lines

def get_tokens(lines, token='word'):
    if token == 'word':
        return [l.split(' ') for l in lines]
    elif token == 'char':
        return [list(l) for l in lines]
    else:
        raise ValueError('Unknown token type: ' + token)

def load_data_wiki(batch_size, max_len):
    print('加载数据...')
    data_dir = os.path.join("data", "wikitext-2")
    lines = read_wiki(data_dir)
    # 构建词表
    tokens = get_tokens(lines, 'word')
    vocab = d2l.Vocab(tokens, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
    dataset = WikiTextDataset(lines, vocab, max_len)
    data_iter = DataLoader(dataset, batch_size, shuffle=True,
                           num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab

# ------------------ 数据集类 ------------------
class WikiTextDataset(Dataset):
    def __init__(self, lines, vocab, max_len):
        # 句对采样
        self.vocab = vocab
        self.max_len = max_len
        self.max_pred = max(1, round(max_len * 0.15))
        self.all_segments = []
        self.is_next = []
        self._prepare_segments(lines)

    def _prepare_segments(self, lines):
        # 采样句对，50%为真实下一句，50%为随机下一句
        segments = []
        for line in lines:
            segs = line.split('. ')
            if len(segs) < 2:
                continue
            segments += segs
        num_segments = len(segments)
        i = 0
        while i < num_segments - 1:
            if random.random() < 0.5:
                # 真实下一句
                self.all_segments.append((segments[i], segments[i+1]))
                self.is_next.append(1)
                i += 2
            else:
                # 随机下一句
                rand_idx = random.randint(0, num_segments-1)
                if rand_idx == i:
                    rand_idx = (rand_idx + 1) % num_segments
                self.all_segments.append((segments[i], segments[rand_idx]))
                self.is_next.append(0)
                i += 1

    def _truncate_pair(self, tokens_a, tokens_b, max_len):
        while len(tokens_a) + len(tokens_b) > max_len - 3:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def __getitem__(self, idx):
        a, b = self.all_segments[idx]
        tokens_a = a.split(' ')
        tokens_b = b.split(' ')
        tokens_a, tokens_b = self._truncate_pair(tokens_a, tokens_b, self.max_len)
        tokens = ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
        segments = ([0] * (len(tokens_a) + 2)) + ([1] * (len(tokens_b) + 1))
        # 转id
        input_ids = self.vocab[tokens]
        # padding
        padding_len = self.max_len - len(input_ids)
        input_ids += [self.vocab['<pad>']] * padding_len
        segments += [0] * padding_len
        # MLM遮盖
        (mlm_input_ids, pred_positions, mlm_labels) = self._get_mlm(input_ids)
        # 补齐或截断 pred_positions 和 mlm_labels
        if len(pred_positions) < self.max_pred:
            pred_positions += [0] * (self.max_pred - len(pred_positions))
            mlm_labels += [0] * (self.max_pred - len(mlm_labels))
        else:
            pred_positions = pred_positions[:self.max_pred]
            mlm_labels = mlm_labels[:self.max_pred]
        return (torch.tensor(mlm_input_ids), torch.tensor(segments),
                torch.tensor(pred_positions), torch.tensor(mlm_labels),
                torch.tensor(self.is_next[idx]), torch.tensor(input_ids))

    def _get_mlm(self, input_ids):
        # 只mask非特殊/pad token
        cand_pos = []
        for i, token_id in enumerate(input_ids):
            token = self.vocab.to_tokens(token_id)
            if token != '<cls>' and token != '<sep>' and token != '<pad>':
                cand_pos.append(i)
        num_pred = max(1, int(round(len(input_ids) * 0.15)))
        random.shuffle(cand_pos)
        pred_positions = cand_pos[:num_pred]
        mlm_input_ids = list(input_ids)
        mlm_labels = [0] * len(pred_positions)
        for i, pos in enumerate(pred_positions):
            mlm_labels[i] = input_ids[pos]
            prob = random.random()
            if prob < 0.8:
                mlm_input_ids[pos] = self.vocab['<mask>']
            elif prob < 0.9:
                mlm_input_ids[pos] = random.randint(0, len(self.vocab)-1)
            # 否则保持原词
        return mlm_input_ids, pred_positions, mlm_labels

    def __len__(self):
        return len(self.all_segments)

# ------------------ BERT模型相关 ------------------
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=512, norm_shape=768):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block{i}',
                d2l.EncoderBlock(
                    # key_size, query_size, value_size, num_hiddens,
                    # norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                    # dropout, use_bias
                    num_hiddens, num_hiddens, num_hiddens, num_hiddens,
                    [num_hiddens], num_hiddens, ffn_num_hiddens, num_heads,
                    dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments) + \
            self.pos_embedding[:, :tokens.shape[1], :]
        X = self.dropout(X)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        # X: (batch, seq_len, hidden)
        num_pred = pred_positions.shape[1]
        batch_size = X.shape[0]
        pred_positions = pred_positions.reshape(-1)
        batch_idx = torch.arange(0, batch_size).repeat_interleave(num_pred)
        # 取出预测位置的编码
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape(batch_size, num_pred, -1)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.linear = nn.Linear(num_hiddens, 2)

    def forward(self, X):
        # X: (batch, seq_len, hidden)
        # 取cls位置
        return self.linear(X[:, 0, :])

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=512, norm_shape=768):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len, norm_shape)
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred(num_hiddens)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X)
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# ------------------ 辅助函数 ------------------
def get_bert_encoding(net, tokens_a, tokens_b=None, vocab=None, device=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    token_ids = vocab[tokens]
    tokens_tensor = torch.tensor(token_ids, device=device).unsqueeze(0)
    segments_tensor = torch.tensor(segments, device=device).unsqueeze(0)
    valid_len = torch.tensor([len(token_ids)], device=device)
    encoded_X = net.encoder(tokens_tensor, segments_tensor, valid_len)
    return encoded_X

# ------------------ 训练函数 ------------------
def train_bert(train_iter, net, loss, vocab, device, num_steps):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    timer = d2l.Timer()
    num_steps = min(num_steps, len(train_iter))
    mlm_losses, nsp_losses = [], []
    step = 0
    timer.start()
    for batch in train_iter:
        mlm_X, segments, pred_positions, mlm_Y, nsp_Y, tokens = [x.to(device) for x in batch]
        valid_lens = (tokens != vocab['<pad>']).sum(dim=1)
        optimizer.zero_grad()
        encoded_X, mlm_Y_hat, nsp_Y_hat = net(mlm_X, segments, valid_lens, pred_positions)
        # MLM loss
        mlm_l = loss(mlm_Y_hat.reshape(-1, mlm_Y_hat.shape[-1]), mlm_Y.reshape(-1))
        # NSP loss
        nsp_l = nn.CrossEntropyLoss()(nsp_Y_hat, nsp_Y)
        l = mlm_l + nsp_l
        l.backward()
        optimizer.step()
        mlm_losses.append(float(mlm_l))
        nsp_losses.append(float(nsp_l))
        step += 1
        if step % 10 == 0 or step == num_steps:
            print(f"step {step}, MLM loss {mlm_l:.4f}, NSP loss {nsp_l:.4f}")
        if step >= num_steps:
            break
    timer.stop()
    print(f"训练总耗时: {timer.sum():.2f} 秒")
    d2l.set_figsize()
    d2l.plt.plot(mlm_losses, label='MLM loss')
    d2l.plt.plot(nsp_losses, label='NSP loss')
    d2l.plt.xlabel('batch')
    d2l.plt.ylabel('loss')
    d2l.plt.legend()
    d2l.plt.title('BERT Pretraining Loss')
    d2l.plt.show()

# ------------------ main 演示部分 ------------------
def main():
    # 1. 加载数据
    batch_size, max_len = 8, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    device = get_device()
    print(f"使用设备: {device}")

    # 2. 演示 BERTEncoder
    num_hiddens, ffn_num_hiddens, num_heads, num_layers = 96, 96, 4, 2
    dropout = 0.2
    net = BERTEncoder(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, max_len, num_hiddens)
    for batch in train_iter:
        tokens_X, segments_X, pred_positions_X, mlm_Y, nsp_Y, tokens = [x.to(device) for x in batch]
        valid_lens = (tokens != vocab['<pad>']).sum(dim=1)
        out = net(tokens_X, segments_X, valid_lens)
        print('BERTEncoder输出 shape:', out.shape)
        break

    # 3. 演示 MaskLM 和 NextSentencePred
    masklm = MaskLM(len(vocab), num_hiddens)
    nsp = NextSentencePred(num_hiddens)
    for batch in train_iter:
        tokens_X, segments_X, pred_positions_X, mlm_Y, nsp_Y, tokens = [x.to(device) for x in batch]
        valid_lens = (tokens != vocab['<pad>']).sum(dim=1)
        encoded_X = net(tokens_X, segments_X, valid_lens)
        mlm_out = masklm(encoded_X, pred_positions_X)
        nsp_out = nsp(encoded_X)
        print('MaskLM输出 shape:', mlm_out.shape)
        print('NSP输出 shape:', nsp_out.shape)
        break

    # 4. 构建 BERTModel 并前向
    bert = BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, max_len, num_hiddens)
    for batch in train_iter:
        tokens_X, segments_X, pred_positions_X, mlm_Y, nsp_Y, tokens = [x.to(device) for x in batch]
        valid_lens = (tokens != vocab['<pad>']).sum(dim=1)
        encoded_X, mlm_Y_hat, nsp_Y_hat = bert(tokens_X, segments_X, valid_lens, pred_positions_X)
        print('BERTModel编码 shape:', encoded_X.shape)
        print('BERTModel MLM输出 shape:', mlm_Y_hat.shape)
        print('BERTModel NSP输出 shape:', nsp_Y_hat.shape)
        break

    # 5. 训练（少量步骤）
    print('开始训练...')
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, bert, loss, vocab, device, num_steps=30)

    # 6. 演示 get_bert_encoding
    test_tokens_a = ['this', 'movie', 'is', 'great']
    test_tokens_b = ['i', 'like', 'it']
    encoding = get_bert_encoding(bert, test_tokens_a, None, vocab, device)
    print('单句编码 shape:', encoding.shape)
    print('单句编码部分值:', encoding[0, :5, :5].detach().cpu().numpy())
    encoding_pair = get_bert_encoding(bert, test_tokens_a, test_tokens_b, vocab, device)
    print('句对编码 shape:', encoding_pair.shape)
    print('句对编码部分值:', encoding_pair[0, :5, :5].detach().cpu().numpy())

if __name__ == '__main__':
    main()