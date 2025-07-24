import torch
x = torch.arange(4.0)           # tensor([0., 1., 2., 3.])
x.requires_grad_(True)          # 告诉 PyTorch：x 需要求梯度
print(x)

y = 2 * torch.dot(x, x)         # y = 2 * (0^2 + 1^2 + 2^2 + 3^2) = 28
y.backward()                    # 自动反向传播，计算 dy/dx

print(x.grad)                   # tensor
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)  # 随机生成一个标量，要求梯度
print(a)
d = f(a)                                      # 执行函数 f
d.backward()                                  # 对最终结果 d 进行反向传播

print(a.grad == d / a)