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

def main():
    # 读取数据集
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 参数设置
    vocab_size, num_hiddens = len(vocab), 256
    device = get_device()
    print(f"Using device: {device}")

    num_epochs, lr = 500, 1

    # 定义 GRU 层
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)

    # 构建模型
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)

    # 训练并计时
    start_time = time.time()
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # 展示所有绘图
    d2l.plt.show()

if __name__ == "__main__":
    main()
