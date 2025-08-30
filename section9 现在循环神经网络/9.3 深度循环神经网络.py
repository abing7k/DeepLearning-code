import time
import torch
from torch import nn
from d2l import torch as d2l


def get_device():
    """自动检测设备: 优先CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    # ======================
    # 1. 加载数据集
    # ======================
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # ======================
    # 2. 定义超参数
    # ======================
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = get_device()
    print(f"Using device: {device}")

    # ======================
    # 3. 定义多层LSTM模型
    # ======================
    # num_layers=2 表示两层堆叠的LSTM
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = d2l.RNNModel(lstm_layer, len(vocab))
    model = model.to(device)

    # ======================
    # 4. 训练与预测
    # ======================
    num_epochs, lr = 500, 2
    start_time = time.time()
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
    end_time = time.time()

    # ======================
    # 5. 打印训练耗时
    # ======================
    print(f"\nTotal training time: {end_time - start_time:.2f} sec")

    # ======================
    # 6. 显示绘制的困惑度曲线图
    # ======================
    d2l.plt.show()


if __name__ == "__main__":
    main()
