import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

def main():
    # 保存和加载单个张量
    x = torch.arange(4)
    torch.save(x, 'x-file')
    x2 = torch.load('x-file')
    print(x2)  # 输出: tensor([0, 1, 2, 3])

    # 保存和加载张量列表
    y = torch.zeros(4)
    torch.save([x, y], 'x-files')
    x2, y2 = torch.load('x-files')
    print((x2, y2))  # 输出: (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))

    # 保存和加载字典
    mydict = {'x': x, 'y': y}
    torch.save(mydict, 'mydict')
    mydict2 = torch.load('mydict')
    print(mydict2)  # 输出: {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}

    # 定义模型并保存参数
    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    torch.save(net.state_dict(), 'mlp.params')

    # 加载模型参数
    clone = MLP()
    clone.load_state_dict(torch.load('mlp.params'))
    clone.eval()
    print(clone)  # 输出模型结构

    # 验证
    Y_clone = clone(X)
    print(Y_clone == Y)  # 输出: tensor([[True, ...], [True, ...]])

if __name__ == "__main__":
    main()