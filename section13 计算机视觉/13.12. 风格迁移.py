# -*- coding: utf-8 -*-
"""
13.12 风格迁移（Style Transfer）

实现步骤：
- 读取内容图像与风格图像并展示；
- 使用预训练 VGG19 抽取内容/风格特征；
- 构建内容损失、风格损失、全变分损失；
- 迭代更新合成图像，输出最终结果。

整理代码、统一 main() 入口，训练阶段输出耗时。所有图像展示调用 d2l.plt.show()。
"""
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from d2l import torch as d2l


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


rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape: Tuple[int, int]) -> torch.Tensor:
    """对输入图像进行缩放、标准化，并添加批量维。"""
    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return transform(img).unsqueeze(0)


def postprocess(img: torch.Tensor):
    """反标准化并转换为 PIL 图像，便于展示。"""
    img = img[0].cpu() * rgb_std[:, None, None] + rgb_mean[:, None, None]
    img = torch.clamp(img, 0, 1)
    return transforms.ToPILImage()(img)


# ---------------------------
# 特征抽取
# ---------------------------
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]


def build_feature_extractor() -> nn.Sequential:
    """裁剪 VGG19 只保留需要的卷积层。"""
    weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
    pretrained_net = torchvision.models.vgg19(weights=weights).features
    for param in pretrained_net.parameters():
        param.requires_grad = False

    max_idx = max(style_layers + content_layers)
    net = nn.Sequential(*[pretrained_net[i] for i in range(max_idx + 1)])
    return net


def extract_features(net: nn.Sequential,
                     X: torch.Tensor,
                     content_idxs: List[int],
                     style_idxs: List[int]):
    """逐层前向传播，收集指定内容层与风格层的输出。"""
    contents, styles = [], []
    for i, layer in enumerate(net):
        X = layer(X)
        if i in style_idxs:
            styles.append(X)
        if i in content_idxs:
            contents.append(X)
    return contents, styles


def get_contents(net, content_img, image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(net, content_X, content_layers, style_layers)
    return content_X, [Y.detach() for Y in contents_Y]


def get_styles(net, style_img, image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(net, style_X, content_layers, style_layers)
    return style_X, [Y.detach() for Y in styles_Y]


# ---------------------------
# 损失函数
# ---------------------------
def gram(X: torch.Tensor) -> torch.Tensor:
    """计算风格特征的 Gram 矩阵。"""
    c = X.shape[1]
    n = X.numel() // c
    X = X.reshape(c, n)
    return torch.matmul(X, X.t()) / (c * n)


def content_loss(Y_hat: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(Y_hat, Y)


def style_loss(Y_hat: torch.Tensor, gram_Y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gram(Y_hat), gram_Y)


def tv_loss(X: torch.Tensor) -> torch.Tensor:
    """全变分损失，抑制高频噪声。"""
    loss_h = torch.abs(X[:, :, 1:, :] - X[:, :, :-1, :]).mean()
    loss_w = torch.abs(X[:, :, :, 1:] - X[:, :, :, :-1]).mean()
    return 0.5 * (loss_h + loss_w)


def compute_loss(X: torch.Tensor,
                 contents_Y_hat,
                 styles_Y_hat,
                 contents_Y,
                 styles_Y_gram,
                 content_weight: float,
                 style_weight: float,
                 tv_weight: float):
    contents_l = [
        content_loss(Y_hat, Y) * content_weight
        for Y_hat, Y in zip(contents_Y_hat, contents_Y)
    ]
    styles_l = [
        style_loss(Y_hat, Y) * style_weight
        for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)
    ]
    tv_l = tv_loss(X) * tv_weight
    total_loss = sum(contents_l) + sum(styles_l) + tv_l
    return contents_l, styles_l, tv_l, total_loss


# ---------------------------
# 合成图像模型
# ---------------------------
class SynthesizedImage(nn.Module):
    """合成图像作为唯一需要训练的参数。"""

    def __init__(self, img_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


def get_inits(X: torch.Tensor,
              styles_Y,
              device: torch.device,
              lr: float):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img, styles_Y_gram, trainer


# ---------------------------
# 训练循环
# ---------------------------
def train_style_transfer(net: nn.Sequential,
                         content_img,
                         style_img,
                         image_shape: Tuple[int, int],
                         device: torch.device,
                         num_epochs: int = 500,
                         lr: float = 0.3,
                         lr_decay_epoch: int = 50,
                         content_weight: float = 1.0,
                         style_weight: float = 1e3,
                         tv_weight: float = 10.0):
    """执行风格迁移训练，返回最终合成图像。"""
    content_X, contents_Y = get_contents(net, content_img, image_shape, device)
    _, styles_Y = get_styles(net, style_img, image_shape, device)
    gen_img, styles_Y_gram, trainer = get_inits(content_X, styles_Y, device, lr)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, gamma=0.8)

    history = {'content': [], 'style': [], 'tv': []}
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        trainer.zero_grad()
        X = gen_img()
        contents_Y_hat, styles_Y_hat = extract_features(
            net, X, content_layers, style_layers
        )
        contents_l, styles_l, tv_l, total_l = compute_loss(
            X, contents_Y_hat, styles_Y_hat,
            contents_Y, styles_Y_gram,
            content_weight, style_weight, tv_weight
        )
        total_l.backward()
        trainer.step()
        scheduler.step()

        history['content'].append(float(sum(contents_l)))
        history['style'].append(float(sum(styles_l)))
        history['tv'].append(float(tv_l))

        if epoch % 50 == 0 or epoch == num_epochs:
            print(
                f'epoch {epoch:03d}: '
                f'content {history["content"][-1]:.3f}, '
                f'style {history["style"][-1]:.3f}, '
                f'tv {history["tv"][-1]:.3f}, '
                f'total {float(total_l):.3f}'
            )

    total_time = time.time() - start_time
    print(f'Total training time: {total_time:.2f} sec on {device}')
    return gen_img().detach()


# ---------------------------
# main() 入口
# ---------------------------
def main():
    d2l.set_figsize()
    content_img = d2l.Image.open('../img/rainier.jpg')
    d2l.plt.imshow(content_img)
    d2l.plt.axis('off')
    d2l.plt.show()

    style_img = d2l.Image.open('../img/autumn-oak.jpg')
    d2l.plt.imshow(style_img)
    d2l.plt.axis('off')
    d2l.plt.show()

    device = get_preferred_device()
    print(f'Running style transfer on device: {device}')

    net = build_feature_extractor().to(device).eval()
    image_shape = (300, 450)
    output = train_style_transfer(
        net=net,
        content_img=content_img,
        style_img=style_img,
        image_shape=image_shape,
        device=device,
        num_epochs=500,
        lr=0.3,
        lr_decay_epoch=50
    )

    result_img = postprocess(output)
    d2l.set_figsize()
    d2l.plt.imshow(result_img)
    d2l.plt.axis('off')
    d2l.plt.show()


if __name__ == '__main__':
    main()
