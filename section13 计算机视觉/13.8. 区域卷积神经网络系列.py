# -*- coding: utf-8 -*-
"""
区域卷积神经网络（R-CNN）系列代码示例。
整理输出，保持原始示例思路，加上 main()、运行耗时打印，
所有图片使用 d2l.plt.show() 方便在 PyCharm 中直接运行。
"""
import time

import torch
import torchvision
from d2l import torch as d2l


def get_dummy_feature_map() -> torch.Tensor:
    """构造 4×4 单通道特征图，重现书中的示例。"""
    X = torch.arange(16., dtype=torch.float32).reshape(1, 1, 4, 4)
    print('[示例特征图 X]')
    print(X)
    return X


def get_example_rois(spatial_scale: float = 1.0) -> torch.Tensor:
    """返回书中使用的两个兴趣区域提议。"""
    if spatial_scale == 1.0:
        rois = torch.tensor([
            [0, 0, 0, 3, 3],
            [0, 1, 1, 4, 4],
        ], dtype=torch.float32)
    else:
        rois = torch.tensor([
            [0, 0, 0, 20, 20],
            [0, 0, 10, 30, 30],
        ], dtype=torch.float32)
    print(f'[兴趣区域 rois | spatial_scale={spatial_scale}]')
    print(rois)
    return rois


def visualize_roi_pool_regions(X: torch.Tensor) -> None:
    """利用热力图展示 ROI Pooling 示例对应的最大池化窗口。"""
    d2l.set_figsize((4, 3))
    ax = d2l.plt.imshow(X.squeeze(0).squeeze(0).numpy(), cmap='viridis')
    d2l.plt.colorbar(ax, fraction=0.046, pad=0.04)
    d2l.plt.title('4×4 Feature Map')
    d2l.plt.tight_layout()
    d2l.plt.show()


def demo_roi_pool_without_scaling() -> torch.Tensor:
    """重现书中对左上角 3×3 区域执行 2×2 ROI Pooling 的输出。"""
    X = get_dummy_feature_map()
    visualize_roi_pool_regions(X)
    rois = get_example_rois(spatial_scale=1.0)
    pooled = torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=1.0)
    print('[ROI Pooling 输出 | 无缩放]')
    print(pooled)
    return pooled


def demo_roi_pool_with_scaling() -> torch.Tensor:
    """演示在特征图与原图分辨率不同情况下的 ROI Pooling。"""
    X = get_dummy_feature_map()
    rois = get_example_rois(spatial_scale=0.1)
    pooled = torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
    print('[ROI Pooling 输出 | spatial_scale=0.1]')
    print(pooled)
    return pooled


def demo_roi_align(X: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """使用 ROI Align（Mask R-CNN 中采用）比较插值效果。"""
    aligned = torchvision.ops.roi_align(X, rois, output_size=(2, 2), spatial_scale=1.0, aligned=True)
    print('[ROI Align 输出]')
    print(aligned)
    return aligned


def main() -> None:
    start = time.time()

    pooled_without_scale = demo_roi_pool_without_scaling()
    pooled_with_scale = demo_roi_pool_with_scaling()

    # 为 Mask R-CNN 示例准备：使用第一个 ROI 做 ROI Align 对比。
    X = get_dummy_feature_map()
    rois = get_example_rois(spatial_scale=1.0)
    _ = demo_roi_align(X, rois[:1])

    # ---- 注释：这里打印整体脚本耗时，满足训练/运行时间输出要求 ----
    print('[脚本耗时]')
    print(f'Total elapsed time: {time.time() - start:.2f} sec')

    # 避免未使用变量的告警，且方便在 REPL 中复用结果。
    return pooled_without_scale, pooled_with_scale


if __name__ == '__main__':
    main()
