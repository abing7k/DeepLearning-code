"""
13.2 微调 (Fine-tuning)
Hotdog 数据集微调 ResNet18
保持书中原始逻辑，适合在 PyCharm 中运行
"""

import os
import time
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


def get_devices():
    """自动检测设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():  # Apple Silicon
        return [torch.device('mps')]
    else:
        return [torch.device('cpu')]


def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True, data_dir=None,
                      train_augs=None, test_augs=None):
    """
    微调训练函数
    param_group=True：除最后一层外使用较小学习率
    """
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                         transform=test_augs),
        batch_size=batch_size)

    devices = get_devices()
    print(f"使用设备: {devices}")

    loss = nn.CrossEntropyLoss(reduction="none")

    if param_group:
        # 输出层以外的参数使用默认学习率
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD(
            [{'params': params_1x},
             {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)

    # 开始计时
    start = time.time()
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer,
                   num_epochs, devices)
    print(f"训练耗时: {time.time() - start:.2f} 秒\n")


def main():
    # 1. 下载数据集
    d2l.DATA_HUB['hotdog'] = (
        d2l.DATA_URL + 'hotdog.zip',
        'fba480ffa8aa7e0febbb511d181409f899b9baa5')
    data_dir = d2l.download_extract('hotdog')
    print(f"数据集路径: {data_dir}")

    # 2. 读取训练/测试集图像
    train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

    # 3. 展示前 8 个正样本与后 8 个负样本
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    d2l.plt.show()

    # 4. 图像增广与标准化
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize])

    # 5. 预训练模型
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    print("\n预训练模型的输出层结构:")
    print(pretrained_net.fc)  # Linear(in_features=512, out_features=1000, bias=True)

    # 6. 微调模型：保留特征层，仅替换输出层
    finetune_net = torchvision.models.resnet18(pretrained=True)
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
    nn.init.xavier_uniform_(finetune_net.fc.weight)

    # 7. 微调训练
    print("\n开始微调训练 (预训练参数 + 小学习率)")
    train_fine_tuning(finetune_net, learning_rate=5e-5,
                      data_dir=data_dir,
                      train_augs=train_augs, test_augs=test_augs)

    # 8. 从零开始训练对比
    print("开始从零训练 (随机初始化 + 大学习率)")
    scratch_net = torchvision.models.resnet18()
    scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
    train_fine_tuning(scratch_net, learning_rate=5e-4,
                      param_group=False,
                      data_dir=data_dir,
                      train_augs=train_augs, test_augs=test_augs)


if __name__ == "__main__":
    main()
