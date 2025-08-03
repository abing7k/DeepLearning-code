import torch
from d2l import torch as d2l  # 确保已通过 pip 安装 d2l： pip install d2l==1.0.3
import matplotlib.pyplot as plt


# # 定义输入张量 x，并启用梯度跟踪
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
#
# # 计算 sigmoid 函数的输出
# y = torch.sigmoid(x)
#
# # 对 y 求导（因为 y 是标量向量，需要提供一个相同形状的梯度）
# y.backward(torch.ones_like(x))
#
# # 绘图：显示 sigmoid 函数和其梯度
# d2l.plot(x.detach().numpy(),
#          [y.detach().numpy(), x.grad.numpy()],
#          legend=['sigmoid', 'gradient'],
#          figsize=(4.5, 2.5))
#
# plt.show()
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))
    print(i)
    print(M)
print('乘以100个矩阵后\n', M)

