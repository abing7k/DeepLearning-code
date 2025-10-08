import torch
import matplotlib.pyplot as plt


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(5, 5), cmap="Reds"):
    """
    显示注意力权重热力图
    :param matrices: 形状 (num_rows, num_cols, num_queries, num_keys)
    :param xlabel: 横轴标签（Keys）
    :param ylabel: 纵轴标签（Queries）
    :param titles: 每个子图的标题（可选）
    :param figsize: 图像大小
    :param cmap: 颜色映射
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])

    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


def main():
    # 构造注意力权重矩阵
    # 单位矩阵：query 只关注与自己相同的 key
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))

    # 可视化
    show_heatmaps(attention_weights,
                  xlabel="Keys",
                  ylabel="Queries")


if __name__ == "__main__":
    main()