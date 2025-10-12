# -*- coding: utf-8 -*-
"""
13.13 实战 Kaggle 比赛：图像分类 (CIFAR-10)

整理代码，保持原始思路不乱改：
- 加上 main()，并在展示图像后执行 d2l.plt.show()，方便在 PyCharm 直接运行
- 保留教材中的输出示例，例如数据集规模统计与训练指标打印
- 对训练时间打印添加注释，便于快速定位训练性能信息
"""
import collections
import math
import os
import shutil
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils import data as torch_data
import torchvision
from torchvision import transforms
from d2l import torch as d2l


# ---------------------------
# Kaggle CIFAR-10 数据准备
# ---------------------------
d2l.DATA_HUB['cifar10_tiny'] = (
    d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
    '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd'
)


def read_csv_labels(fname: str) -> Dict[str, str]:
    """读取标签 CSV，返回文件名到标签的映射字典。"""
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过表头
    tokens = [line.rstrip().split(',') for line in lines]
    return {name: label for name, label in tokens}


def copyfile(filename: str, target_dir: str) -> None:
    """将文件复制到目标目录（若不存在则创建）。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir: str,
                      labels: Dict[str, str],
                      valid_ratio: float) -> int:
    """将验证集从原始训练集中拆分出来。"""
    counter = collections.Counter(labels.values())
    n = counter.most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count: Dict[str, int] = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(
            data_dir, 'train_valid_test', 'train_valid', label))
        if label_count.get(label, 0) < n_valid_per_label:
            copyfile(fname, os.path.join(
                data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(
                data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label


def reorg_test(data_dir: str) -> None:
    """在预测阶段整理测试集目录结构，方便读取。"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(
            os.path.join(data_dir, 'test', test_file),
            os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')
        )


def reorg_cifar10_data(data_dir: str, valid_ratio: float) -> int:
    """综合整理 Kaggle CIFAR-10 数据集。"""
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
    return len(labels)


# ---------------------------
# 模型与训练逻辑
# ---------------------------
def get_net() -> nn.Module:
    """构建 ResNet-18 模型，类别数为 10。"""
    num_classes = 10
    return d2l.resnet18(num_classes, in_channels=3)


loss = nn.CrossEntropyLoss(reduction="none")


def try_all_gpus() -> List[torch.device]:
    """尝试获取全部可用 GPU，否则回退到 MPS 或 CPU。"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if devices:
        return devices
    if torch.backends.mps.is_available():
        return [torch.device('mps')]
    return [torch.device('cpu')]


def prepare_net_for_training(net: nn.Module,
                             devices: List[torch.device]) -> nn.Module:
    """根据可用设备配置模型并返回可训练网络。"""
    primary = devices[0]
    if primary.type == 'cuda' and len(devices) > 1:
        device_ids = list(range(len(devices)))
        net = nn.DataParallel(net, device_ids=device_ids)
        return net.to(primary)
    if primary.type in ('cuda', 'mps'):
        return net.to(primary)
    return net.to(primary)


def train(net: nn.Module,
          train_iter: torch_data.DataLoader,
          valid_iter: torch_data.DataLoader,
          num_epochs: int,
          lr: float,
          wd: float,
          devices: List[torch.device],
          lr_period: int,
          lr_decay: float) -> Tuple[nn.Module, d2l.Animator]:
    """训练模型，可选使用验证集进行评估。"""
    trainer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        trainer, step_size=lr_period, gamma=lr_decay
    )
    num_batches = len(train_iter)
    timer = d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)

    net = prepare_net_for_training(net, devices)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels, loss, trainer, devices
            )
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            epoch_fraction = epoch + (i + 1) / num_batches
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if valid_iter is not None:
                animator.add(epoch_fraction, (train_loss, train_acc, None))
            else:
                animator.add(epoch_fraction, (train_loss, train_acc))

        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
        measures = (
            f'train loss {metric[0] / metric[2]:.3f}, '
            f'train acc {metric[1] / metric[2]:.3f}'
        )
        if valid_iter is not None:
            measures += f', valid acc {valid_acc:.3f}'
        # 打印训练过程中的性能指标及吞吐量估计，便于对照书中结果
        print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f} '
              f'examples/sec on {devices}')
    print(f'Total training time: {timer.sum():.2f} sec on {devices[0]}')
    d2l.plt.show()
    return net, animator


# ---------------------------
# 工具函数与演示逻辑
# ---------------------------
def show_sample_images(data_dir: str,
                       labels: Dict[str, str],
                       indices: Iterable[int],
                       rows: int = 2,
                       cols: int = 4,
                       scale: float = 1.2) -> None:
    """展示部分训练图片及其标签示例。"""
    images = []
    sample_labels: List[str] = []
    for idx in indices:
        file_name = f'{idx}.png'
        path = os.path.join(data_dir, 'train', file_name)
        if not os.path.exists(path):
            continue
        image = d2l.Image.open(path)
        images.append(image)
        sample_labels.append(labels[str(idx)])
    if images:
        d2l.set_figsize()
        d2l.show_images(images, rows, cols, scale=scale)
        d2l.plt.show()
        print('示例标签:', sample_labels)


def build_dataloaders(data_dir: str,
                      batch_size: int) -> Tuple[
                          torch_data.DataLoader,
                          torch_data.DataLoader,
                          torch_data.DataLoader,
                          torch_data.DataLoader]:
    """根据整理后的数据目录构建数据加载器。"""
    transform_train = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_train
        ) for folder in ['train', 'train_valid']
    ]
    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_test
        ) for folder in ['valid', 'test']
    ]

    train_iter, train_valid_iter = [
        torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=d2l.get_dataloader_workers()
        ) for dataset in (train_ds, train_valid_ds)
    ]
    valid_iter = torch_data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=d2l.get_dataloader_workers()
    )
    test_iter = torch_data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=d2l.get_dataloader_workers()
    )
    return train_iter, train_valid_iter, valid_iter, test_iter


def generate_submission(net: nn.Module,
                        test_iter: torch_data.DataLoader,
                        train_valid_ds: torchvision.datasets.ImageFolder,
                        devices: List[torch.device],
                        output_path: str = 'submission.csv') -> pd.DataFrame:
    """使用训练完成的模型对 Kaggle 测试集生成提交文件。"""
    preds: List[int] = []
    primary = devices[0]
    net.eval()
    with torch.no_grad():
        for X, _ in test_iter:
            X = X.to(primary)
            y_hat = net(X)
            preds.extend(
                y_hat.argmax(dim=1).type(torch.int32).cpu().numpy().tolist()
            )
    sorted_ids = list(range(1, len(test_iter.dataset) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
    df.to_csv(output_path, index=False)
    print(f'Kaggle 提交文件已生成：{output_path}')
    return df


# ---------------------------
# main 演示流程
# ---------------------------
def main():
    demo = True  # 若使用完整 Kaggle 数据集，将该变量改为 False
    if demo:
        data_dir = d2l.download_extract('cifar10_tiny')
    else:
        data_dir = '../data/cifar-10/'

    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    print('# 训练样本 :', len(labels))
    print('# 类别 :', len(set(labels.values())))

    show_sample_images(data_dir, labels, indices=range(1, 9))

    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, valid_ratio)

    batch_size = 32 if demo else 128
    train_iter, train_valid_iter, valid_iter, test_iter = build_dataloaders(
        data_dir, batch_size
    )

    devices = try_all_gpus()
    num_epochs, lr, wd = 20, 2e-4, 5e-4
    lr_period, lr_decay = 4, 0.9
    print(f'Using devices: {devices}')
    net = get_net()
    net, animator = train(
        net,
        train_iter,
        valid_iter,
        num_epochs,
        lr,
        wd,
        devices,
        lr_period,
        lr_decay
    )

    print('开始使用全部标注数据重新训练模型...')
    net, _ = train(
        get_net(),
        train_valid_iter,
        None,
        num_epochs,
        lr,
        wd,
        devices,
        lr_period,
        lr_decay
    )

    generate_submission(net, test_iter, train_valid_iter.dataset, devices)


if __name__ == '__main__':
    main()
