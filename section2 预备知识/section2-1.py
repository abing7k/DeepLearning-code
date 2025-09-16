import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.size())
print(x.numel())

x = x.reshape(3, 4)
print(x)

x = torch.zeros(2, 3, 4)
print(x)

x = torch.ones(2, 3, 4)
print(x)

x = torch.randn(3, 4)
print(x)

x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(x)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

print(X == Y)
print(X.sum())

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a, b)
print(a + b)

x = torch.arange(12).reshape((3, 4))
print(x[-1], x[1:3])

print(x)
x[0:2, :] = 12
print(x)

print("torch.arange(3,4)",torch.arange(12).reshape((3,4)))
z=torch.zeros_like(Y)
print(z,"  id(z)",id(z))
z[:]=x+y
print(z,"  id(z)",id(z))
print("z,x+y",z,x+y,z==x+y)

a=x.numpy()
b=torch.tensor(a)
print(type(a),type(b))

a=torch.tensor(3.5)
print(a,a.item(),float(a),int(a))