import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),  # net[0]
    nn.ReLU(),        # net[1]
    nn.Linear(8, 1)   # net[2]
)

X = torch.rand(size=(2, 4))
print(net(X))

# 打印每一层的参数
print("【net[0].state_dict()】：第一层 Linear(4, 8) 的参数")
print(net[0].state_dict())

print("【net[1].state_dict()】：第二层 ReLU，无参数")
print(net[1].state_dict())

print("【net[2].state_dict()】：第三层 Linear(8, 1) 的参数")
print(net[2].state_dict())

# 查看偏置的类型和值
print("【type(net[2].bias)】：偏置的类型")
print(type(net[2].bias))

print("【net[2].bias】：偏置本身（包含梯度信息的张量）")
print(net[2].bias)

print("【net[2].bias.data】：偏置的原始值（不带梯度信息）")
print(net[2].bias.data)


print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)