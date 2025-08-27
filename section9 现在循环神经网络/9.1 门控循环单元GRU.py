import torch
from torch import nn
from d2l import torch as d2l
import time

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_params(vocab_size, num_hiddens, device):
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xz = normal((vocab_size, num_hiddens))
    W_hz = normal((num_hiddens, num_hiddens))
    b_z = torch.zeros(num_hiddens, device=device)

    W_xr = normal((vocab_size, num_hiddens))
    W_hr = normal((num_hiddens, num_hiddens))
    b_r = torch.zeros(num_hiddens, device=device)

    W_xh = normal((vocab_size, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, vocab_size))
    b_q = torch.zeros(vocab_size, device=device)

    params = [W_xz, W_hz, b_z,
              W_xr, W_hr, b_r,
              W_xh, W_hh, b_h,
              W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilda = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

def main():
    # 读取数据集
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 设置参数
    vocab_size, num_hiddens = len(vocab), 256
    device = get_device()
    print(f"Using device: {device}")

    num_epochs, lr = 500, 1

    # 构建模型
    model = d2l.RNNModelScratch(
        vocab_size, num_hiddens, device, get_params,
        init_gru_state, gru)

    # 训练并计时
    start_time = time.time()
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # 确保所有绘图展示
    d2l.plt.show()

if __name__ == "__main__":
    main()
