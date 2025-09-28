# -*- coding: utf-8 -*-
"""
动手学深度学习 v2 13.1 图像增广
整理版：适用于 PyCharm
"""
import time
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


# -----------------------------
# 辅助函数
# -----------------------------
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    """多次应用增广并显示结果"""
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()


def load_cifar10(is_train, augs, batch_size):
    """加载 CIFAR-10 并应用图像增广"""
    dataset = torchvision.datasets.CIFAR10(
        root="../data", train=is_train,
        transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """单批次训练（支持多GPU/单GPU/CPU）"""
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer,
               num_epochs, devices):
    """完整训练（支持多GPU/单GPU/CPU）"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(
        xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
        legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)  # 训练损失, 准确度, 样本数, 特征数
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2],
                              metric[1] / metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch {epoch+1}, loss {metric[0]/metric[2]:.3f}, '
              f'train acc {metric[1]/metric[3]:.3f}, '
              f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {devices}')
    print(f'Total training time: {timer.sum():.2f} sec')
    d2l.plt.show()


def train_with_data_aug(train_augs, test_augs, net,
                        batch_size=256, lr=0.001, num_epochs=10):
    """封装完整训练流程"""
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    devices = try_all_gpus()
    print(f"Using devices: {devices}")
    train_ch13(net, train_iter, test_iter, loss, trainer,
               num_epochs, devices)




def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if devices:     # 有CUDA则直接返回
        return devices
    # 2️⃣ 如果没有 CUDA，则检测 MPS
    if torch.backends.mps.is_available():
        return [torch.device('mps')]

    # 3️⃣ 都没有则使用 CPU
    return [torch.device('cpu')]
# -----------------------------
# 主流程
# -----------------------------
def main():

    # ===== 13.1.2 使用图像增广进行训练 =====
    # 展示 CIFAR-10 前32张图像
    all_images = torchvision.datasets.CIFAR10(
        train=True, root="./data", download=True)
    d2l.show_images([all_images[i][0] for i in range(32)],
                    4, 8, scale=0.8)
    d2l.plt.show()

    # 设置图像显示大小
    d2l.set_figsize()
    img = d2l.Image.open('../img/cat1.jpg')
    d2l.plt.imshow(img)
    d2l.plt.show()

    # # ===== 13.1.1 常用增广方法 =====
    # # 左右翻转
    # apply(img, torchvision.transforms.RandomHorizontalFlip())
    # # 上下翻转
    # apply(img, torchvision.transforms.RandomVerticalFlip())
    # # 随机裁剪并缩放到200x200
    # shape_aug = torchvision.transforms.RandomResizedCrop(
    #     (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # apply(img, shape_aug)
    # # 改变亮度
    # apply(img, torchvision.transforms.ColorJitter(
    #     brightness=0.5, contrast=0, saturation=0, hue=0))
    # # 改变色调
    # apply(img, torchvision.transforms.ColorJitter(
    #     brightness=0, contrast=0, saturation=0, hue=0.5))
    # # 同时改变多项颜色属性
    # color_aug = torchvision.transforms.ColorJitter(
    #     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # apply(img, color_aug)
    # # 组合多种增广
    # augs = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(),
    #     color_aug, shape_aug])
    # apply(img, augs)

    # 仅训练集使用增广：随机左右翻转 + ToTensor
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    # 定义ResNet18
    batch_size = 256
    net = d2l.resnet18(10, 3)
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    start = time.time()
    train_with_data_aug(train_augs, test_augs, net,
                        batch_size=batch_size, lr=0.001, num_epochs=10)
    print(f"Total script time: {time.time() - start:.2f} sec")


# -----------------------------
if __name__ == '__main__':
    main()
