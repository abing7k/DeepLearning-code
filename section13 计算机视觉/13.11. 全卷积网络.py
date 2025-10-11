# -*- coding: utf-8 -*-
"""
13.11 全卷积网络（Fully Convolutional Networks，FCN）

整理示例包括：
1. 利用转置卷积对图像进行上采样演示；
2. 构建基于预训练 ResNet18 的 FCN 结构；
3. 训练并评估 VOC2012 语义分割数据集；
4. 对测试图像进行像素级预测与可视化。

- 统一入口 main()
- 训练阶段使用自定义设备检测并输出耗时
- 图像展示统一调用 d2l.plt.show()，便于在 PyCharm 直接运行
"""
import os
import time

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF
from d2l import torch as d2l
import chapter13_voc as voc_module


# ---------------------------
# 通用工具
# ---------------------------
def get_preferred_device() -> torch.device:
    """选择可用的训练设备，优先 CUDA -> MPS -> CPU。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def import_voc_module():
    """动态加载 13.9 章节的 VOC 数据处理脚本。"""
    return voc_module


# ---------------------------
# 13.11.2 转置卷积与双线性插值
# ---------------------------
def bilinear_kernel(in_channels: int, out_channels: int,
                    kernel_size: int) -> torch.Tensor:
    """构造双线性插值权重，用于转置卷积初始化。"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = torch.arange(kernel_size, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor)
    kernel = filt.unsqueeze(0) * filt.unsqueeze(1)

    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    for i in range(in_channels):
        weight[i, i] = kernel
    return weight


def demo_transposed_conv_image() -> None:
    """演示转置卷积对猫狗图片做 2 倍上采样的效果。"""
    img = transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
    conv_trans = nn.ConvTranspose2d(
        in_channels=3,
        out_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        bias=False
    )
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

    X = img.unsqueeze(0)
    Y = conv_trans(X)
    out_img = Y[0].permute(1, 2, 0).detach()

    d2l.set_figsize()
    print('input image shape:', img.permute(1, 2, 0).shape)
    d2l.plt.imshow(img.permute(1, 2, 0))
    d2l.plt.axis('off')
    d2l.plt.show()

    print('output image shape:', out_img.shape)
    d2l.plt.imshow(out_img)
    d2l.plt.axis('off')
    d2l.plt.show()


# ---------------------------
# 13.11.2 FCN 架构定义
# ---------------------------
class FullyConvNet(nn.Module):
    """基于 ResNet18 骨干的全卷积网络。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        pretrained_net = torchvision.models.resnet18(weights=weights)

        # 去除最后两个模块（自适应池化与全连接层）
        self.features = nn.Sequential(*list(pretrained_net.children())[:-2])
        self.head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.transpose_conv = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, padding=16, stride=32
        )

        # 参数初始化：1x1 卷积使用 Xavier，转置卷积使用双线性插值
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.transpose_conv.weight.data.copy_(
            bilinear_kernel(num_classes, num_classes, 64)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.features(X)
        X = self.head(X)
        X = self.transpose_conv(X)
        return X


# ---------------------------
# 13.11.4 训练流程
# ---------------------------
def segmentation_loss(inputs: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """像素级交叉熵损失，按小批量样本维度返回均值。"""
    losses = F.cross_entropy(inputs, targets, reduction='none')
    return losses.mean(dim=1).mean(dim=1)


def train_fcn(net: nn.Module,
              train_iter,
              test_iter,
              device: torch.device,
              num_epochs: int,
              lr: float,
              weight_decay: float) -> None:
    """训练全卷积网络并输出每轮指标与耗时。"""
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.to(device)

    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        train_correct_pixels = 0.0
        total_pixels = 0

        for features, labels in train_iter:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = net(features)
            loss_vals = segmentation_loss(preds, labels)
            loss = loss_vals.mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss_vals.sum().item()
            train_sample_count += loss_vals.numel()
            train_correct_pixels += (
                (preds.argmax(dim=1) == labels).float().sum().item()
            )
            total_pixels += labels.numel()

        avg_train_loss = train_loss_sum / train_sample_count
        train_acc = train_correct_pixels / total_pixels
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(
            f'epoch {epoch + 1:02d}: loss {avg_train_loss:.3f}, '
            f'train acc {train_acc:.3f}, test acc {test_acc:.3f}'
        )

    total_time = time.time() - start_time
    examples_per_sec = (
        len(train_iter.dataset) * num_epochs / total_time if total_time > 0 else 0.0
    )
    print(f'{examples_per_sec:.1f} examples/sec on {device}')
    print(f'Total training time: {total_time:.2f} sec')


def evaluate_accuracy(net: nn.Module, data_iter, device: torch.device) -> float:
    """在验证集上计算像素级准确率。"""
    net.eval()
    correct_pixels = 0.0
    total_pixels = 0
    with torch.no_grad():
        for features, labels in data_iter:
            features = features.to(device)
            labels = labels.to(device)
            preds = net(features)
            correct_pixels += (preds.argmax(dim=1) == labels).float().sum().item()
            total_pixels += labels.numel()
    net.train()
    return correct_pixels / total_pixels


# ---------------------------
# 13.11.5 预测与可视化
# ---------------------------
def predict(net: nn.Module,
            dataset,
            img: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    """对输入图像（未归一化）执行预测并返回类别索引。"""
    net.eval()
    with torch.no_grad():
        X = dataset.normalize_image(img).unsqueeze(0).to(device)
        pred = net(X).argmax(dim=1)
    return pred.squeeze(0).cpu()


def label2image(pred: torch.Tensor, colormap) -> torch.Tensor:
    """将类别索引映射回 VOC 的 RGB 颜色。"""
    color_tensor = torch.tensor(colormap, dtype=torch.uint8)
    X = pred.long()
    return color_tensor[X]


# ---------------------------
# main() 入口
# ---------------------------
def main():
    demo_transposed_conv_image()

    voc_module = import_voc_module()
    num_classes = len(voc_module.VOC_CLASSES)

    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = voc_module.load_data_voc(batch_size, crop_size)

    device = get_preferred_device()
    print(f'Training on device: {device}')

    net = FullyConvNet(num_classes)
    train_fcn(
        net=net,
        train_iter=train_iter,
        test_iter=test_iter,
        device=device,
        num_epochs=5,
        lr=0.001,
        weight_decay=1e-3
    )

    # 预测示例
    voc_dir = d2l.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    test_images, test_labels = voc_module.read_voc_images(voc_dir, False)

    n = 4
    imgs = []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = TF.crop(test_images[i], *crop_rect)
        pred_indices = predict(net, test_iter.dataset, X, device)
        pred_color = label2image(pred_indices, voc_module.VOC_COLORMAP)
        label_crop = TF.crop(test_labels[i], *crop_rect)

        imgs.append((X.permute(1, 2, 0) / 255).cpu())
        imgs.append((pred_color.float() / 255))
        imgs.append((label_crop.permute(1, 2, 0) / 255).cpu())

    d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
    d2l.plt.show()


if __name__ == '__main__':
    main()
