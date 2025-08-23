# rnn_from_scratch_d2l.py
#
# Minimal, faithful reformatting of the code you provided from the d2l (Dive into Deep Learning) book.
# I kept the algorithm and function names unchanged, only added:
#  - a `main()` entry point
#  - device detection (CUDA / MPS / CPU)
#  - training time measurements
#  - explicit plt.show() after plotting so images appear in PyCharm
#  - clear comments and small I/O prints for shape / example outputs
#
# Run this file in PyCharm (or any Python environment with the `d2l` package installed).


import math
import time
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from d2l import torch as d2l


# ---------------------- Utility: device detection -------------------------
def get_device():
    """Detect an available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # PyTorch MPS backend (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------- One-hot example & shape checks -------------------
# (kept from original code; useful sanity checks printed in main)


# ---------------------- Parameter initialization -------------------------

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# ---------------------- RNN state and forward ----------------------------

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


# ---------------------- Model wrapper (from-scratch) --------------------
class RNNModelScratch:  # @save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # X is expected with shape: (batch_size, num_steps)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# ---------------------- Prediction helper -------------------------------

def predict_ch8(prefix, num_preds, net, vocab, device):  # @save
    """在prefix后面生成新字符（按原书实现）"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# ---------------------- Gradient clipping -------------------------------

def grad_clipping(net, theta):  # @save
    """裁剪梯度，接受从零实现的模型或nn.Module"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ---------------------- Training helpers --------------------------------
# train_epoch_ch8 与 train_ch8 基本沿用原书实现，加入了训练速度和timing打印

# @save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# @save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章），并返回最后困惑度和词元/秒"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    ppl = None
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(f'[epoch {epoch+1}] sample prediction: {predict("time traveller")}')
            animator.add(epoch + 1, [ppl])
    # 确保图像在PyCharm等环境中显示
    try:
        plt.show()
    except Exception:
        pass
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print('Final predictions:')
    print(predict('time traveller'))
    print(predict('traveller'))
    return ppl, speed


# ---------------------- Main: put everything together -------------------

def main():
    device = get_device()
    print('Detected device:', device)

    # Basic dataset and data loader as in original code
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # One-hot examples / shape checks from original notebook
    print('\nOne-hot demo and shape checks:')
    print('F.one_hot([0,2]) =>', F.one_hot(torch.tensor([0, 2]), len(vocab)))
    X = torch.arange(10).reshape((2, 5))
    print('one_hot(X.T, 28).shape =>', F.one_hot(X.T, 28).shape)

    # 初始化并测试模型输出形状
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                          init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], device)
    Y, new_state = net(X.to(device), state)
    print('\nShape checks for model outputs:')
    print('Y.shape =>', Y.shape)
    print('len(new_state) =>', len(new_state))
    print('new_state[0].shape =>', new_state[0].shape)

    # 训练（按原始参数）：num_epochs=500, lr=1
    num_epochs, lr = 500, 1
    print('\nStart training (sequential partitioning)')
    start = time.time()
    ppl, speed = train_ch8(net, train_iter, vocab, lr, num_epochs, device)
    duration = time.time() - start
    print(f'Training finished in {duration:.1f} seconds ({duration/60:.2f} minutes).')

    # 随机抽样训练，重新初始化模型
    print('\nStart training (random sampling) — reinitializing the network')
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                          init_rnn_state, rnn)
    start = time.time()
    ppl2, speed2 = train_ch8(net, train_iter, vocab, lr, num_epochs, device,
                             use_random_iter=True)
    duration2 = time.time() - start
    print(f'Random-iter training finished in {duration2:.1f} seconds ({duration2/60:.2f} minutes).')

    print('\nAll done.')

if __name__ == '__main__':
    main()
