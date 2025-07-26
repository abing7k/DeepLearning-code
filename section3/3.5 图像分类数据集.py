import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from d2l import torch as d2l


# -------------------------------
# 数据预处理和标签可视化函数


# 加载 Fashion-MNIST 数据集
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中

    参数：
        batch_size: 每个批次(batch)包含的图像数量
        resize: 可选参数，如果给定，例如 resize=64，则将图像缩放为64x64

    返回：
        (train_iter, test_iter): 分别为训练集和测试集的 DataLoader（迭代器）
    """

    # 创建一个列表，用于存储图像的预处理操作
    # transforms.ToTensor()：把图像从 PIL 格式转换为 PyTorch 的张量（tensor）格式
    # 同时会将像素值从 [0, 255] 范围转换为 [0.0, 1.0] 的 float32 类型
    trans = [transforms.ToTensor()]

    # 如果传入了 resize 参数（如 resize=64），表示我们想把图像缩放成 64x64 大小
    if resize:
        # 插入 Resize 操作到转换列表最前面（先缩放，再转为 tensor）
        trans.insert(0, transforms.Resize(resize))

    # 将多个图像变换操作（如 Resize + ToTensor）组合成一个整体流程
    trans = transforms.Compose(trans)

    # 下载并加载训练集：
    # - root="../data"：数据保存到 ../data 文件夹中
    # - train=True 表示是训练集
    # - transform=trans 表示应用我们上面组合的图像预处理操作
    # - download=True 表示如果本地没有数据，就自动从网上下载
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)

    # 下载并加载测试集，参数和上面类似，只是 train=False 表示是测试数据
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    # 返回训练数据和测试数据的 DataLoader
    # DataLoader 是 PyTorch 中用于“批量读取数据”的工具
    # - batch_size：一次读取多少张图片
    # - shuffle=True 表示打乱数据顺序（用于训练更有效）
    # - num_workers=get_dataloader_workers() 表示使用多少个子进程并行读取（加速）
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def get_fashion_mnist_labels(labels):
    """将数字标签转换为文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.squeeze().numpy(), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


# -------------------------------
# 主函数放到 if __name__ == "__main__"
if __name__ == "__main__":

    train_iter, test_iter = load_data_fashion_mnist(batch_size=32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
    # 显示部分图像
    # batch_size_show = 18
    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=batch_size_show)))
    # show_images(X.reshape(batch_size_show, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    # # 测试 DataLoader 加载时间
    # batch_size = 256
    # train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
    #                              num_workers=get_dataloader_workers())
    #
    # timer = d2l.Timer()
    # for X, y in train_iter:
    #     continue
    # print(f'{timer.stop():.2f} sec')
