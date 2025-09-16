import torch

x = torch.arange(4)
print(x)
print(len(x))
print(x.shape)
print(torch.arange(12).reshape(3, 4).shape)

A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B.T)
print(B == B.T)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)
print((a * X).shape)

print(A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)
print(A.mean(), A.sum() / A.numel())

print(A.sum(axis=0, keepdims=True))
print(A)
print(A.cumsum(axis=0))

x = torch.tensor([0., 1., 2., 3.])
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))

print("矩阵-向量积")
print(x)
print(A)

print(A.shape)
print(x.shape)
print(torch.mv(A, x))

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
print(A)
print(B)
print(torch.mm(A, B))

u = torch.tensor([3., -4.])
print(torch.norm(u))
print(torch.abs(u).sum())


u1 = torch.arange(12.).reshape(3,4)
print(u1)
print("范数")
print(torch.norm(u1, p=1))
print(torch.norm(u1, p=2))
print(torch.norm(u1, p=3))


import torch

# 设置随机种子以确保结果可重复
torch.manual_seed(42)

# 定义示例矩阵 A 和 B（2x2 矩阵）
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print("1. 证明一个矩阵A的转置的转置是A，即(A^T)^T = A。")
print("A =\n", A)
A_transpose = A.T
A_transpose_transpose = A_transpose.T
print("(A^T)^T =\n", A_transpose_transpose)
print("是否等于 A:", torch.allclose(A, A_transpose_transpose))

print("\n2. 给出两个矩阵A和B，证明 '它们转置的和' 等于 '它们和的转置'，即A^T + B^T = (A + B)^T。")
print("A =\n", A)
print("B =\n", B)
A_plus_B = A + B
A_plus_B_transpose = A_plus_B.T
A_transpose_plus_B_transpose = A.T + B.T
print("(A + B)^T =\n", A_plus_B_transpose)
print("A^T + B^T =\n", A_transpose_plus_B_transpose)
print("是否等于:", torch.allclose(A_plus_B_transpose, A_transpose_plus_B_transpose))

print("\n3. 给定任意矩阵A，A + A^T 总是对称的吗?为什么?")
print("A =\n", A)
A_plus_A_transpose = A + A.T
print("A + A^T =\n", A_plus_A_transpose)
print("是否对称 (A + A^T == (A + A^T)^T):", torch.allclose(A_plus_A_transpose, A_plus_A_transpose.T))
print("解释: 对称矩阵满足 A = A^T，A + A^T 总是对称的，因为 (A + A^T)^T = A^T + A = A + A^T。")

print("\n4. 本节中定义了形状(2, 3, 4)的张量X。1eN(X)的输出结果是什么?")
X = torch.arange(12).reshape(1, 3, 4).float()  # 形状 (2, 3, 4) 的张量
print("X =\n", X)
print(len(X))
print("1eN(X) 含义不明确，假设为 len(X.shape) 或 X.norm()，这里取 X 的维度数:", len(X.shape))

print("\n5. 对于任意形状的张量X,len(X)返回X的张量对应于x特定轴的长度?这个轴是什么?")
print("X.shape =", X.shape)
print("len(X) =", len(X), "表示最外层维度 (axis=0) 的长度，即", X.shape[0])

print("\n6. 运行A/A.sum(axis=1)，看看会发生什么。请分析一下原因?")
print("A =\n", A)
A_sum_axis1 = A.sum(axis=1, keepdim=True)  # 沿着 axis=1 求和，保持维度
print("A.sum(axis=1, keepdim=True) =\n", A_sum_axis1)
A_div_sum = A / A_sum_axis1
print("A / A.sum(axis=1) =\n", A_div_sum)
print("原因: A.sum(axis=1) 沿着列求和，结果是 [3, 7] (1x2 向量)，直接除会维度不匹配。使用 keepdim=True 确保维度一致，结果为沿列归一化的矩阵。")

print("\n7. 考虑一个具有形状(2, 3, 4)的张量，在轴0、1、2上的求和输出是什么?")
print("X.shape =", X.shape)
print("sum over axis 0:\n", X.sum(dim=0))
print("sum over axis 1:\n", X.sum(dim=1))
print("sum over axis 2:\n", X.sum(dim=2))

print("\n8. 对于 linalg.norm函数提供3个或更多维度的张量，并观察其输出。对手写形状的张量这个函数计算什么?")


# 设置随机种子以确保结果可重复
torch.manual_seed(42)

# 创建 3D 张量 (2, 3, 4)
X_3d = torch.arange(24).reshape(2, 3, 4).float()  # 确保浮点类型
print("3D 张量 X_3d:")
print(X_3d)
print("X_3d.shape =", X_3d.shape)

# 测试 3D 张量的情况
print("\n3D 张量范数计算:")
try:
    print("默认 ord=2 (无 dim):", torch.linalg.norm(X_3d, ord=2))  # 应报错
except RuntimeError as e:
    print("错误:", e)

# 使用 torch.norm 计算 Frobenius 范数
print("Frobenius 范数 (ord='fro'):", torch.norm(X_3d, p='fro').item())

print("沿 dim=0 的向量 L2 范数:", torch.linalg.norm(X_3d, ord=2, dim=0))
print("沿 dim=(0, 1) 的矩阵谱范数:", torch.linalg.norm(X_3d, ord=2, dim=(0, 1)))

# 创建 4D 张量 (2, 3, 4, 5)
X_4d = torch.arange(120).reshape(2, 3, 4, 5).float()  # 确保浮点类型
print("\n4D 张量 X_4d:")
print(X_4d)
print("X_4d.shape =", X_4d.shape)

# 测试 4D 张量的情况
print("\n4D 张量范数计算:")
try:
    print("默认 ord=2 (无 dim):", torch.linalg.norm(X_4d, ord=2))  # 应报错
except RuntimeError as e:
    print("错误:", e)

print("Frobenius 范数 (ord='fro'):", torch.norm(X_4d, p='fro').item())
print("沿 dim=(2, 3) 的矩阵谱范数:", torch.linalg.norm(X_4d, ord=2, dim=(2, 3)))

# 解释
print("\n解释:")
print("对于任意形状的张量，torch.linalg.norm 计算向量或矩阵范数，具体取决于 ord 和 dim 参数：")
print("- 默认 (ord=None, dim=None): 展平为 1D 向量，计算 L2 范数。")
print("- ord=2 需指定 dim: 整数为向量 L2 范数，2-tuple 为矩阵谱范数。")
print("- ord='fro': 计算 Frobenius 范数 (所有元素平方和的平方根)，支持高维张量，但需用 torch.norm。")
print("高维张量视为批次处理，dim 指定计算维度，剩余维度保持。")