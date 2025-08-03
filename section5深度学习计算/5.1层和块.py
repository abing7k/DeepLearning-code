import torch
from torch import nn
from torch.nn import functional as F


net = nn.Sequential(
    nn.Linear(20, 256),  # 全连接层1：输入20维 -> 输出256维
    nn.ReLU(),           # ReLU激活函数
    nn.Linear(256, 10)   # 全连接层2：输入256维 -> 输出10维
)


X = torch.rand(2, 20)  # 创建一个输入张量
output = net(X)        # 执行前向传播
print(output)          # ✅ 打印输出结果


# 自定义块：多层感知机（MLP）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层：输入20维 -> 输出256维
        self.out = nn.Linear(256, 10)     # 输出层：输入256维 -> 输出10维

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))  # 使用ReLU激活函数

# 测试MLP的前向传播
def main():
    net = MLP()  # 实例化模型
    X = torch.rand(2, 20)  # 构造一个2个样本、每个样本20维的输入张量
    output = net(X)  # 前向传播计算输出
    print(output)  # 打印模型输出

if __name__ == "__main__":
    main()
