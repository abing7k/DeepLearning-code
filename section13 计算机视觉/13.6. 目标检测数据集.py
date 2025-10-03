# 13.6 目标检测数据集（香蕉检测）
# 保持原始逻辑 + main() + 时间打印 + 只展示前10张图像
# 可直接在 PyCharm 中运行

import os
import time
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l


# ---------------------------
# 13.6.1 下载数据集
# ---------------------------
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)


# ---------------------------
# 13.6.2 读取数据集
# ---------------------------
#@save
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(
        data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv'
    )
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')

    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir,
                         'bananas_train' if is_train else 'bananas_val',
                         'images', f'{img_name}')))
        # target: (类别，左上角x，左上角y，右下角x，右下角y)
        # 数据集只有一个类别（香蕉，索引为0）
        targets.append(list(target))

    # 返回图像和标签（归一化到 0~1）
    return images, torch.tensor(targets).unsqueeze(1) / 256


#@save
class BananasDataset(torch.utils.data.Dataset):
    """自定义香蕉检测数据集"""
    def __init__(self, is_train):
        start = time.time()
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) +
              (' training examples' if is_train else ' validation examples'))
        print(f"加载数据用时: {time.time() - start:.2f} 秒")

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


#@save
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(
        BananasDataset(is_train=False), batch_size)
    return train_iter, val_iter


# ---------------------------
# 13.6.3 演示（只展示10张）
# ---------------------------
def show_10_images(batch, edge_size=256):
    """展示小批量中的前10张图像及其边界框"""
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = d2l.show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, batch[1][0:10]):
        d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
    d2l.plt.show()


# ---------------------------
# main() 入口
# ---------------------------
def main():
    batch_size, edge_size = 32, 256
    train_iter, val_iter = load_data_bananas(batch_size)

    # ---- 示例：打印一个小批量的形状 ----
    batch = next(iter(train_iter))
    print("训练批量图像形状:", batch[0].shape)
    print("训练批量标签形状:", batch[1].shape)

    # ---- 示例：展示前10张图像 + 边界框 ----
    print("\n展示训练批量中的前10张图像及其边界框...")
    show_10_images(batch, edge_size)


if __name__ == '__main__':
    main()