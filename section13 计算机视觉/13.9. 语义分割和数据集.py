# -*- coding: utf-8 -*-
"""
13.9 语义分割和数据集

整理自《动手学深度学习》相关章节，保留原始示例结构：
- 统一入口 main()
- 图像展示均调用 d2l.plt.show()
- 输出示例与书中保持一致
"""
import os
import torch
import torchvision
from torchvision.transforms import functional as TF
from d2l import torch as d2l


# ---------------------------
# 13.9.2 Pascal VOC2012 语义分割数据集
# ---------------------------
#@save
d2l.DATA_HUB['voc2012'] = (
    d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
    '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'
)


#@save
def read_voc_images(voc_dir: str, is_train: bool = True):
    """读取所有 VOC 图像及其语义分割标签。"""
    txt_fname = os.path.join(
        voc_dir, 'ImageSets', 'Segmentation',
        'train.txt' if is_train else 'val.txt'
    )
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r', encoding='utf-8') as f:
        images = f.read().split()

    features, labels = [], []
    for fname in images:
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')
            )
        )
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'),
                mode
            )
        )
    return features, labels


#@save
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]


#@save
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]


#@save
def voc_colormap2label():
    """构建从 RGB 值到 VOC 类别索引的查找表。"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for idx, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        ] = idx
    return colormap2label


#@save
def voc_label_indices(colormap: torch.Tensor, colormap2label: torch.Tensor):
    """将标签图中的 RGB 颜色转换为类别索引。"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = (
        (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
        + colormap[:, :, 2]
    )
    return colormap2label[idx]


# ---------------------------
# 预处理工具
# ---------------------------
#@save
def voc_rand_crop(feature: torch.Tensor,
                  label: torch.Tensor,
                  height: int,
                  width: int):
    """对特征和标签执行同步随机裁剪。"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width)
    )
    feature = TF.crop(feature, *rect)
    label = TF.crop(label, *rect)
    return feature, label


# ---------------------------
# 自定义数据集
# ---------------------------
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """Pascal VOC 语义分割数据集封装。"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img: torch.Tensor):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [
            img for img in imgs
            if img.shape[1] >= self.crop_size[0]
            and img.shape[2] >= self.crop_size[1]
        ]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size
        )
        return feature, voc_label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)


#@save
def load_data_voc(batch_size, crop_size):
    """加载 VOC 语义分割数据集迭代器。"""
    voc_dir = d2l.download_extract(
        'voc2012', os.path.join('VOCdevkit', 'VOC2012')
    )
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir),
        batch_size,
        drop_last=True,
        num_workers=num_workers
    )
    return train_iter, test_iter


# ---------------------------
# main() 入口
# ---------------------------
def main():
    # 下载并读取 VOC2012 数据集
    voc_dir = d2l.download_extract(
        'voc2012', os.path.join('VOCdevkit', 'VOC2012')
    )
    train_features, train_labels = read_voc_images(voc_dir, True)

    # 展示前 5 个样本及其标签
    n = 5
    imgs = train_features[0:n] + train_labels[0:n]
    imgs = [img.permute(1, 2, 0) for img in imgs]
    d2l.show_images(imgs, 2, n, scale=1.2)
    d2l.plt.show()

    # 映射示例：打印某块区域的类别索引
    cmap2label = voc_colormap2label()
    indices = voc_label_indices(train_labels[0], cmap2label)
    sample_patch = indices[105:115, 130:140]
    print('[类别索引示例补丁]')
    print(sample_patch)
    print('对应类别:', VOC_CLASSES[1])

    # 随机裁剪示例
    crops = []
    for _ in range(n):
        feat, lab = voc_rand_crop(train_features[0], train_labels[0], 200, 300)
        crops.append(feat.permute(1, 2, 0))
        crops.append(lab.permute(1, 2, 0))
    d2l.show_images(crops, 2, n, scale=1.2)
    d2l.plt.show()

    # 构建数据迭代器并查看批量形状
    crop_size = (320, 480)
    batch_size = 64
    train_iter, test_iter = load_data_voc(batch_size, crop_size)
    print('train examples:', len(train_iter.dataset))
    print('test examples:', len(test_iter.dataset))

    first_batch = next(iter(train_iter))
    print('feature batch shape:', first_batch[0].shape)
    print('label batch shape:', first_batch[1].shape)


if __name__ == '__main__':
    main()
