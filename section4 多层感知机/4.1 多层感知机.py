# 4.1.2 激活函数：ReLU 绘图示例（适用于 PyCharm）

import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l


# 创建输入张量，从 -8 到 8，步长为 0.1，并启用梯度计算
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# print(x)


# 计算 ReLU 激活函数输出
y = torch.relu(x)
# print(y)

def showimg():
    # 绘制图像
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    plt.show()

    y.backward(torch.ones_like(x), retain_graph=True)
    print(torch.ones_like(x))
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
    plt.show()



def main():
    showimg()




if __name__ == '__main__':
    main()