# -*- coding: utf-8 -*-
"""
13.14 Kaggle Dog Breed Identification (ImageNet Dogs)

- Organize the workflow in main() for IDE execution
- Keep the original semantics while adding comments, device detection, and timing
- Detect devices manually (cuda/mps/cpu) and report training time
- Use d2l.plt.show() for all figures so they render in PyCharm
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
import torchvision
from torchvision import transforms
from torchvision.models import ResNet34_Weights, resnet34
from d2l import torch as d2l


# ---------------------------
# Data download and organization
# ---------------------------
d2l.DATA_HUB['dog_tiny'] = (
    d2l.DATA_URL + 'kaggle_dog_tiny.zip',
    '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d'
)

loss = nn.CrossEntropyLoss(reduction='none')

# Ensure pretrained weights download into the project data directory
def setup_torch_cache() -> Path:
    """Configure torch hub cache to reside under the repository data folder."""
    cache_root = Path(__file__).resolve().parents[1] / 'data' / 'torch_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TORCH_HOME', str(cache_root))
    os.environ.setdefault('TORCH_MODEL_ZOO', str(cache_root))
    torch.hub.set_dir(str(cache_root))
    return cache_root


TORCH_CACHE_DIR = setup_torch_cache()


def detect_devices() -> List[torch.device]:
    """Return the available compute devices, preferring cuda then mps then cpu."""
    if torch.cuda.is_available():
        return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return [torch.device('mps')]
    return [torch.device('cpu')]


def reorg_dog_data(data_dir: str, valid_ratio: float) -> None:
    """Reorganize the dataset following the D2L dog breed instructions."""
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


def show_sample_images(data_dir: str,
                       file_names: Sequence[str],
                       rows: int = 2,
                       cols: int = 4,
                       scale: float = 1.2) -> None:
    """Display sample images from the dataset."""
    images = []
    for name in file_names:
        path = os.path.join(data_dir, 'train', name)
        if os.path.exists(path):
            images.append(d2l.Image.open(path))
    if images:
        d2l.set_figsize()
        d2l.show_images(images, rows, cols, scale=scale)
        d2l.plt.show()


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create the train/test augmentation pipelines."""
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform_train, transform_test


def build_dataloaders(data_dir: str,
                      batch_size: int,
                      transform_train: transforms.Compose,
                      transform_test: transforms.Compose) -> Tuple[
                          torch_data.DataLoader,
                          torch_data.DataLoader,
                          torch_data.DataLoader,
                          torch_data.DataLoader]:
    """Construct DataLoader instances for train/valid/test splits."""
    train_ds, train_valid_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_train
        ) for folder in ('train', 'train_valid')
    ]
    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train_valid_test', folder),
            transform=transform_test
        ) for folder in ('valid', 'test')
    ]
    train_iter, train_valid_iter = [
        torch_data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=d2l.get_dataloader_workers()
        ) for dataset in (train_ds, train_valid_ds)
    ]
    valid_iter = torch_data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, drop_last=True,
        num_workers=d2l.get_dataloader_workers()
    )
    test_iter = torch_data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=d2l.get_dataloader_workers()
    )
    return train_iter, train_valid_iter, valid_iter, test_iter


# ---------------------------
# Model and training
# ---------------------------
class FinetuneResNet34(nn.Module):
    """Fine-tune a pretrained ResNet-34 with a custom classifier head."""

    def __init__(self, num_classes: int = 120) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT
        base_model = resnet34(weights=weights)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.output_new = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.features(x)
        return self.output_new(feature_maps)


def prepare_model_for_devices(net: nn.Module, devices: Sequence[torch.device]) -> nn.Module:
    """Move the model to the primary device and wrap with DataParallel when possible."""
    primary = devices[0]
    net = net.to(primary)
    if len(devices) > 1 and primary.type == 'cuda':
        device_ids = [device.index for device in devices if device.type == 'cuda']
        net = nn.DataParallel(net, device_ids=device_ids)
    return net


def evaluate_loss(data_iter: torch_data.DataLoader,
                  net: nn.Module,
                  device: torch.device) -> torch.Tensor:
    """Evaluate the mean cross-entropy loss on the validation set."""
    net.eval()
    metric_sum, count = 0.0, 0
    with torch.no_grad():
        for features, labels in data_iter:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            l = loss(outputs, labels)
            metric_sum += float(l.sum())
            count += labels.numel()
    net.train()
    return torch.tensor(metric_sum / count, device=device)


def train(net: nn.Module,
          train_iter: torch_data.DataLoader,
          valid_iter: Optional[torch_data.DataLoader],
          num_epochs: int,
          lr: float,
          wd: float,
          devices: Sequence[torch.device],
          lr_period: int,
          lr_decay: float) -> Tuple[nn.Module, d2l.Animator]:
    """Train the custom classifier head and monitor losses."""
    net = prepare_model_for_devices(net, devices)
    trainer = torch.optim.SGD(
        (param for param in net.parameters() if param.requires_grad),
        lr=lr, momentum=0.9, weight_decay=wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=lr_period, gamma=lr_decay)

    num_batches = len(train_iter)
    timer = d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for batch_idx, (features, labels) in enumerate(train_iter):
            timer.start()
            features = features.to(devices[0])
            labels = labels.to(devices[0])
            trainer.zero_grad()
            outputs = net(features)
            l = loss(outputs, labels).sum()
            l.backward()
            trainer.step()
            metric.add(float(l), labels.shape[0])
            timer.stop()

            epoch_fraction = epoch + (batch_idx + 1) / num_batches
            train_loss = metric[0] / metric[1]
            if valid_iter is not None:
                animator.add(epoch_fraction, (train_loss, None))
            else:
                animator.add(epoch_fraction, (train_loss,))

        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices[0])
            animator.add(epoch + 1, (None, float(valid_loss)))
            measures += f', valid loss {float(valid_loss):.3f}'
        scheduler.step()
        throughput = metric[1] * (epoch + 1) / timer.sum()
        print(f'{measures}\n{throughput:.1f} examples/sec on {devices}')

    total_time = timer.sum()
    print(f'Total training time: {total_time:.2f} sec on {devices[0]}')
    d2l.plt.show()
    return net, animator


# ---------------------------
# Inference and submission
# ---------------------------
def generate_submission(net: nn.Module,
                        test_iter: torch_data.DataLoader,
                        class_names: Iterable[str],
                        devices: Sequence[torch.device],
                        output_path: str = 'submission.csv') -> None:
    """Generate a Kaggle submission file from test predictions."""
    primary = devices[0]
    net.eval()
    predictions: List[List[float]] = []
    with torch.no_grad():
        for features, _ in test_iter:
            features = features.to(primary)
            logits = net(features)
            probs = F.softmax(logits, dim=1)
            predictions.extend(probs.cpu().numpy().tolist())

    ids = sorted(os.listdir(os.path.join(test_iter.dataset.root, 'unknown')))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('id,' + ','.join(class_names) + '\n')
        for file_id, row in zip(ids, predictions):
            f.write(file_id.split('.')[0] + ',' + ','.join(str(num) for num in row) + '\n')
    print(f'Kaggle submission saved to {output_path}')


# ---------------------------
# main entry
# ---------------------------
def main():
    demo = True  # Switch to False to train on the full competition dataset
    if demo:
        data_dir = d2l.download_extract('dog_tiny')
    else:
        data_dir = os.path.join('..', 'data', 'dog-breed-identification')

    print('Data directory:', data_dir)

    labels: Dict[str, str] = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    print('Number of train images:', len(labels))
    print('Number of dog breeds:', len(set(labels.values())))
    sample_files = [f'{name}.jpg' for name in list(labels.keys())[:8]]
    show_sample_images(data_dir, sample_files)

    valid_ratio = 0.1
    reorg_dog_data(data_dir, valid_ratio)

    batch_size = 32 if demo else 128
    transform_train, transform_test = build_transforms()
    train_iter, train_valid_iter, valid_iter, test_iter = build_dataloaders(
        data_dir, batch_size, transform_train, transform_test
    )

    devices = detect_devices()
    print('Using devices:', devices)
    num_epochs, lr, wd = 10, 1e-4, 1e-4
    lr_period, lr_decay = 2, 0.9

    trained_net, _ = train(
        FinetuneResNet34(),
        train_iter,
        valid_iter,
        num_epochs,
        lr,
        wd,
        devices,
        lr_period,
        lr_decay
    )

    print('Retraining on the merged train+valid set...')
    trained_net, _ = train(
        FinetuneResNet34(),
        train_valid_iter,
        None,
        num_epochs,
        lr,
        wd,
        devices,
        lr_period,
        lr_decay
    )

    generate_submission(
        trained_net,
        test_iter,
        train_valid_iter.dataset.classes,
        devices
    )


if __name__ == '__main__':
    main()
