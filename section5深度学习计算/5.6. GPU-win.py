import torch
from torch import nn

# 5.6.1 计算设备

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def section_561():
    print("== 5.6.1 计算设备 ==")
    print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
    print("可用GPU数量:", torch.cuda.device_count())
    print("try_gpu():", try_gpu())
    print("try_gpu(10):", try_gpu(10))  # 如果不存在，返回 CPU
    print("try_all_gpus():", try_all_gpus())

# 5.6.2 张量与GPU
def section_562():
    print("\n== 5.6.2 张量与GPU ==")

    x = torch.tensor([1, 2, 3])
    print("x.device:", x.device)

    # 5.6.2.1 存储在GPU上
    X = torch.ones(2, 3, device=try_gpu())
    print("X:", X)

    if torch.cuda.device_count() >= 2:
        Y = torch.rand(2, 3, device=try_gpu(1))
        print("Y:", Y)

        # 5.6.2.2 复制
        Z = X.cuda(1)
        print("X (cuda:0):", X)
        print("Z (cuda:1):", Z)
        print("Y + Z:", Y + Z)
        print("Z.cuda(1) is Z:", Z.cuda(1) is Z)
    else:
        print("此设备没有足够的GPU来演示多个设备之间的复制。")

# 5.6.3 神经网络与GPU
def section_563():
    print("\n== 5.6.3 神经网络与GPU ==")
    device = try_gpu()
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
