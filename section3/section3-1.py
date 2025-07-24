import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt  # 如果你后续需要绘图

from d2l import torch as d2l  # 确保你安装了 d2l 库：pip install d2l


# -------------------------------
# 定义 Timer 类，用于计时
# -------------------------------
class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并记录时间"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


# # -------------------------------
# # 创建两个向量，比较 for 循环 与 矢量加法 的效率
# # -------------------------------
# n = 10000
# a = torch.ones(n)
# b = torch.ones(n)
#
# # 方法1：使用 for 循环执行逐元素相加
# c = torch.zeros(n)
# timer = Timer()
# for i in range(n):
#     c[i] = a[i] + b[i]
# print(f"For 循环耗时: {timer.stop():.5f} sec")
#
# # 方法2：使用矢量化方式相加（推荐方式）
# timer.start()
# d = a + b
# print(f"矢量加法耗时: {timer.stop():.5f} sec")


# 定义正态分布概率密度函数
def normal(x, mu, sigma):
    """
    计算正态分布的概率密度函数值。

    参数：
        x     : 输入值，可以是numpy数组
        mu    : 均值（mean）
        sigma : 标准差（standard deviation）

    返回：
        正态分布的概率密度
    """
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


# 生成 x 的取值范围，从 -7 到 7，步长为 0.01
x = np.arange(-7, 7, 0.01)

# 设置多组 (均值, 标准差) 参数，绘制不同形状的正态分布
params = [(0, 1),  # 标准正态分布，μ=0, σ=1
          (0, 2),  # 宽胖型分布，μ=0, σ=2
          (3, 1)]  # 向右移动的分布，μ=3, σ=1

# 初始化画布
plt.figure(figsize=(8, 4))  # 设置图像大小（英寸）

# 为每组参数画出一条正态分布曲线
for mu, sigma in params:
    y = normal(x, mu, sigma)  # 计算 y 值
    label = f'mean {mu}, std {sigma}'  # 设置图例名称
    plt.plot(x, y, label=label)  # 绘制曲线

# 设置图形标签
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Normal Distribution Curves')
plt.legend()  # 显示图例
plt.grid(True)  # 添加网格
plt.tight_layout()  # 自动调整布局
plt.show()  # 显示图形