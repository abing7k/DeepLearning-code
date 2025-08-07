

import torch
from torch import nn

# 判断是否支持 MPS（Apple Silicon 的 GPU 支持）
def try_mps():
    """如果支持 MPS，则返回 mps 设备，否则返回 cpu"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def section_561():
    print("== 5.6.1 计算设备 ==")
    print("CPU:", torch.device('cpu'))
    print("MPS 是否可用:", torch.backends.mps.is_available())
    print("try_mps():", try_mps())

# 5.6.2 张量与GPU (MPS)
def section_562():
    print("\n== 5.6.2 张量与GPU ==")

    x = torch.tensor([1, 2, 3])
    print("x.device:", x.device)

    # 存储在MPS上（如果可用）
    device = try_mps()
    X = torch.ones(2, 3, device=device)
    print("X.device:", X.device)

# 5.6.3 神经网络与GPU (MPS)
def section_563():
    print("\n== 5.6.3 神经网络与GPU ==")
    device = try_mps()
    net = nn.Sequential(nn.Linear(3, 1))
    net = net.to(device)

    X = torch.ones(2, 3, device=device)
    Y = net(X)
    print("模型输出 Y:", Y)
    print("模型参数所在设备:", net[0].weight.data.device)

def main():
    section_561()
    section_562()
    section_563()

if __name__ == '__main__':
    main()