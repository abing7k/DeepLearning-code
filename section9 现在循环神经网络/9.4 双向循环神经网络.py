import time
import torch
from torch import nn
from d2l import torch as d2l


def get_device():
    """检测可用设备：CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    # 1. 设置参数
    batch_size, num_steps = 32, 35
    device = get_device()
    print(f"Using device: {device}")

    # 2. 加载数据集（Time Machine）
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 3. 定义双向 LSTM 模型
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers,
                         bidirectional=True)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model = model.to(device)

    # 4. 训练模型并计时
    num_epochs, lr = 500, 1
    start_time = time.time()
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    end_time = time.time()

    # 打印训练时间
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds")

    # 5. 显示训练过程中生成的图表
    d2l.plt.show()


if __name__ == "__main__":
    main()
