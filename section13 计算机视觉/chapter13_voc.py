# -*- coding: utf-8 -*-
"""Pascal VOC helper utilities for chapter 13 segmentation examples."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch
import torchvision
from torchvision.transforms import functional as TF
from d2l import torch as d2l

# Keep dataset metadata in the shared D2L registry so download_extract works.
d2l.DATA_HUB['voc2012'] = (
    d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
    '4e443f8a2eca6b1dac8a6c57641b67dd40621a49',
)

# RGB colormap and class labels from the original VOC dataset.
VOC_COLORMAP: List[List[int]] = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
]

VOC_CLASSES: List[str] = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor',
]


def read_voc_images(voc_dir: str, is_train: bool = True):
    """Load VOC feature-label image pairs from disk."""
    txt_fname = os.path.join(
        voc_dir, 'ImageSets', 'Segmentation',
        'train.txt' if is_train else 'val.txt',
    )
    with open(txt_fname, 'r', encoding='utf-8') as f:
        image_names = f.read().split()

    mode = torchvision.io.image.ImageReadMode.RGB
    features, labels = [], []
    for name in image_names:
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{name}.jpg'),
            )
        )
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{name}.png'),
                mode,
            )
        )
    return features, labels


def voc_colormap2label() -> torch.Tensor:
    """Map RGB colors in VOC masks to label indices."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for idx, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        ] = idx
    return colormap2label


def voc_label_indices(colormap: torch.Tensor, colormap2label: torch.Tensor):
    """Convert RGB mask tensor to class index tensor."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    indices = (
        (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
        + colormap[:, :, 2]
    )
    return colormap2label[indices]


def voc_rand_crop(feature: torch.Tensor,
                  label: torch.Tensor,
                  height: int,
                  width: int):
    """Randomly crop feature/label tensors to the target size."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width),
    )
    feature = TF.crop(feature, *rect)
    label = TF.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """Segmentation dataset wrapper for Pascal VOC."""

    def __init__(self, is_train: bool, crop_size: Sequence[int], voc_dir: str):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self._filter(features)]
        self.labels = self._filter(labels)
        self.colormap2label = voc_colormap2label()
        print(f'read {len(self.features)} examples')

    def normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img.float() / 255)

    def _filter(self, imgs):
        return [
            img for img in imgs
            if img.shape[1] >= self.crop_size[0]
            and img.shape[2] >= self.crop_size[1]
        ]

    def __getitem__(self, idx: int):
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size,
        )
        return feature, voc_label_indices(label, self.colormap2label)

    def __len__(self) -> int:
        return len(self.features)


def load_data_voc(batch_size: int, crop_size: Tuple[int, int]):
    """Create DataLoader pairs for Pascal VOC segmentation."""
    voc_dir = d2l.download_extract(
        'voc2012', os.path.join('VOCdevkit', 'VOC2012'),
    )
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir),
        batch_size,
        drop_last=True,
        num_workers=num_workers,
    )
    return train_iter, test_iter


__all__ = [
    'VOC_COLORMAP',
    'VOC_CLASSES',
    'VOCSegDataset',
    'load_data_voc',
    'read_voc_images',
    'voc_colormap2label',
    'voc_label_indices',
    'voc_rand_crop',
]
