import torch
from torch import nn

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()

    # 默认 forward 方法（nn.Module.__call__ 会调用它）
    def forward(self, X, scale=1):
        print("forward 被调用")
        print("X shape:", X.shape)
        print("scale:", scale)
        return X * scale

    # 自定义另一个方法
    def forward2(self, X):
        print("forward2 被调用")
        print("X shape:", X.shape)
        return X + 1

# 创建一个对象
blk = MyBlock()

# 创建输入
X1 = torch.rand(2, 3)

print("\n=== 调用 blk(X1)（自动调用 forward）===")
Y1 = blk(X1)  # 自动调用 forward(X1)
print("Y1:", Y1)

print("\n=== 调用 blk(X1, 0.5)（自动调用 forward，第二个参数传 scale）===")
Y2 = blk(X1, 0.5)  # 自动调用 forward(X1, scale=0.5)
print("Y2:", Y2)

print("\n=== 手动调用 forward2（不会自动调用，需要手动执行）===")
Y3 = blk.forward2(X1)  # 直接手动调用 forward2
print("Y3:", Y3)