# 13.7 单发多框检测（SSD）
# 维持《动手学深度学习》原始实现逻辑，补充 main() 入口、注释及训练时间打印
# 所有图像使用 d2l.plt.show() 显示，保证可在 PyCharm 直接运行

import os
import time
from typing import List, Sequence, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


torch.set_printoptions(2)


def get_device() -> torch.device:
    """自动检测可用设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():  # Apple Silicon
        print('MPS backend detected, falling back to CPU to avoid view/reshape autograd issues.')
        return torch.device('cpu')
    return torch.device('cpu')

def cls_predictor(num_inputs: int, num_anchors: int, num_classes: int) -> nn.Conv2d:
    """类别预测层：通道维存储 a*(q+1) 个类别预测结果"""
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs: int, num_anchors: int) -> nn.Conv2d:
    """边界框预测层：为每个锚框预测 4 个偏移量"""
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred: torch.Tensor) -> torch.Tensor:
    """将通道维移到最后，再展平为二维 (batch, -1)。确保内存连续，兼容 MPS。"""
    t = pred.permute(0, 2, 3, 1)
    return t.reshape(t.size(0), -1)


def concat_preds(preds: Sequence[torch.Tensor]) -> torch.Tensor:
    """串联不同尺度的预测输出"""
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels: int, out_channels: int) -> nn.Sequential:
    """高宽减半块：两个 3x3 卷积 + 批归一化 + ReLU，再接 2x2 最大池化"""
    blk: List[nn.Module] = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net() -> nn.Sequential:
    """基础网络：堆叠 3 个高宽减半块"""
    blk: List[nn.Module] = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i: int) -> nn.Module:
    """根据索引返回 SSD 的第 i 个模块"""
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X: torch.Tensor,
                blk: nn.Module,
                size: Sequence[float],
                ratio: Sequence[float],
                cls_predictor_blk: nn.Module,
                bbox_predictor_blk: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """前向传播：输出特征图、锚框、类别预测与偏移量预测"""
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor_blk(Y)
    bbox_preds = bbox_predictor_blk(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    """章节 13.7 中的 TinySSD 模型"""

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(sizes[0]) + len(ratios[0]) - 1
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], self.num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], self.num_anchors))

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f'blk_{i}'),
                self.sizes[i],
                self.ratios[i],
                getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}')
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks,
              cls_loss, bbox_loss):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_inputs = cls_preds.reshape(-1, num_classes)
    cls_targets = cls_labels.reshape(-1)
    cls_vals = cls_loss(cls_inputs, cls_targets)
    cls = cls_vals.reshape(batch_size, -1).mean(dim=1)

    bbox_inputs = bbox_preds * bbox_masks
    bbox_targets = bbox_labels * bbox_masks
    bbox_vals = bbox_loss(bbox_inputs, bbox_targets)
    bbox = bbox_vals.mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds: torch.Tensor, cls_labels: torch.Tensor) -> float:
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds: torch.Tensor, bbox_labels: torch.Tensor, bbox_masks: torch.Tensor) -> float:
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def demo_shapes() -> None:
    """打印书中各个模块的张量形状示例"""
    print('\n[形状演示]')
    Y1 = cls_predictor(8, 5, 10)(torch.zeros((2, 8, 20, 20)))
    Y2 = cls_predictor(16, 3, 10)(torch.zeros((2, 16, 10, 10)))
    print('Y1 shape:', Y1.shape, 'Y2 shape:', Y2.shape)
    print('concat_preds shape:', concat_preds([Y1, Y2]).shape)
    print('down_sample_blk output:', down_sample_blk(3, 10)(torch.zeros((2, 3, 20, 20))).shape)
    print('base_net output:', base_net()(torch.zeros((2, 3, 256, 256))).shape)


def train_tiny_ssd(net: TinySSD,
                   train_iter: torch.utils.data.DataLoader,
                   device: torch.device,
                   num_epochs: int = 20) -> Tuple[List[float], List[float]]:
    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    timer = d2l.Timer()
    cls_err_history, bbox_mae_history = [], []

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            anchors, cls_preds, bbox_preds = net(X)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks,
                          cls_loss, bbox_loss)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
            
        cls_err = 1 - metric[0] / metric[1]
        bbox_mae = metric[2] / metric[3]
        cls_err_history.append(cls_err)
        bbox_mae_history.append(bbox_mae)
        print(f'epoch {epoch + 1:02d}: class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')

    elapsed_per_example = len(train_iter.dataset) / timer.stop()
    # ---- 训练速度与时间打印 ----
    print(f'{elapsed_per_example:.1f} examples/sec on {device}')
    return cls_err_history, bbox_mae_history


def plot_training_curves(cls_err_history: Sequence[float], bbox_mae_history: Sequence[float]) -> None:
    epochs = range(1, len(cls_err_history) + 1)
    d2l.set_figsize()
    d2l.plt.plot(epochs, cls_err_history, label='class error')
    d2l.plt.plot(epochs, bbox_mae_history, label='bbox mae')
    d2l.plt.xlabel('epoch')
    d2l.plt.legend()
    d2l.plt.grid(True)
    d2l.plt.show()


def predict(net: TinySSD, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


def display_detection(img: torch.Tensor, output: torch.Tensor, threshold: float = 0.9) -> None:
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.numpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, f'{score:.2f}', 'w')
    d2l.plt.axis('off')
    d2l.plt.show()


def main() -> None:
    global_start = time.time()
    demo_shapes()

    device = get_device()
    print(f'Using device: {device}')

    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

    batch_size = 32
    train_iter, _ = d2l.load_data_bananas(batch_size)

    net = net.to(device)
    cls_err_history, bbox_mae_history = train_tiny_ssd(net, train_iter, device)
    plot_training_curves(cls_err_history, bbox_mae_history)

    banana_img_path = '../img/banana.jpg'
    if not os.path.exists(banana_img_path):
        raise FileNotFoundError('Expected banana image at ../img/banana.jpg')
    X = torchvision.io.read_image(banana_img_path).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()
    detections = predict(net, X, device)
    display_detection(img, detections.cpu(), threshold=0.9)

    print(f'Total elapsed time: {time.time() - global_start:.2f} seconds')


if __name__ == '__main__':
    main()
