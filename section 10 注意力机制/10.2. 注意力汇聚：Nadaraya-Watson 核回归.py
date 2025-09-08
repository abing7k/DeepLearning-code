import torch
from d2l import torch as d2l
import time

# 固定使用 CPU
device = torch.device('cpu')
print(f"Using device: {device}")

def main():
    # 生成训练数据集
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = 2 * x_train + 3 * torch.sin(4 * x_train) + torch.normal(0.0, 0.5, (n_train,))

    # 生成测试数据集
    n_test = 100
    x_test = torch.linspace(0, 5, n_test)

    # 高斯核函数
    def gaussian_kernel(x1, x2, bandwidth):
        return torch.exp(-0.5 * ((x1 - x2) / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))

    # Nadaraya-Watson 核回归算法
    def nadaraya_watson(x_train, y_train, x_test, bandwidth):
        weights = gaussian_kernel(x_train.unsqueeze(1), x_test.unsqueeze(0), bandwidth)
        weights = weights / weights.sum(dim=0)
        y_pred = (weights * y_train.unsqueeze(1)).sum(dim=0)
        return y_pred

    # 绘制核回归预测结果
    def plot_kernel_reg(y_pred, bandwidth):
        d2l.plot(x_test, [y_pred, 2 * x_test + 3 * torch.sin(4 * x_test)],
                 'x', 'y', legend=[f'Predicted bw={bandwidth}', 'True'])
        d2l.plt.show()

    # 通过不同的带宽参数观察核回归的拟合效果
    for bandwidth in [0.1, 0.3, 1]:
        y_pred = nadaraya_watson(x_train, y_train, x_test, bandwidth)
        plot_kernel_reg(y_pred, bandwidth)

    # 利用核函数计算权重矩阵并绘制热力图
    bandwidth = 0.3
    weights = gaussian_kernel(x_train.unsqueeze(1), x_train.unsqueeze(0), bandwidth)
    d2l.show_heatmaps(weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted training inputs')
    d2l.plt.show()

    # 计算训练时间
    start = time.time()
    for bandwidth in [0.1, 0.3, 1]:
        y_pred = nadaraya_watson(x_train, y_train, x_test, bandwidth)
    end = time.time()
    print(f"Training time: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()