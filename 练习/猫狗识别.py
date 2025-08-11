

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. 自动检测设备（优先 MPS，然后 CUDA，最后 CPU）
def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS device.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device.")
        return torch.device("cuda")
    else:
        print("Using CPU device.")
        return torch.device("cpu")

# 2. 定义仅包含CIFAR-10猫和狗的数据集
class CatDogCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        # CIFAR10: 3=cat, 5=dog
        indices = [i for i, t in enumerate(self.targets) if t in [3, 5]]
        self.data = self.data[indices]
        self.targets = [self.targets[i] for i in indices]
        # Map 3->0 (cat), 5->1 (dog)
        self.targets = [0 if t == 3 else 1 for t in self.targets]
        self.classes = ['cat', 'dog']

# 3. 可视化若干训练图片及其标签
def imshow(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.axis('off')

def visualize_samples(loader, classes, num_images=8):
    # 获取一批图片
    dataiter = iter(loader)
    images, labels = next(dataiter)
    plt.figure(figsize=(12, 2))
    for idx in range(num_images):
        plt.subplot(1, num_images, idx+1)
        imshow(images[idx], title=classes[labels[idx]])
    plt.suptitle("Sample Training Images")
    plt.show()

# 4. 定义一个简单的全连接神经网络（MLP）模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 2类：猫和狗

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成一维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 5. 训练函数（记录时间）
def train_model(model, device, trainloader, criterion, optimizer, epochs=5):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    total_time = time.time() - start_time
    print(f"Finished Training in {total_time:.2f} seconds.")
    return total_time

# 6. 测试函数，并可视化部分测试结果
def test_model(model, device, testloader, classes, num_visualize=8):
    model.eval()
    correct = 0
    total = 0
    all_images = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 收集用于可视化的图片
            if len(all_images) < num_visualize:
                take = min(num_visualize - len(all_images), images.size(0))
                all_images.append(images[:take].cpu())
                all_labels.extend(labels[:take].cpu().tolist())
                all_preds.extend(predicted[:take].cpu().tolist())
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
    # 可视化部分测试结果
    if all_images:
        images_vis = torch.cat(all_images, dim=0)[:num_visualize]
        plt.figure(figsize=(12, 2))
        for idx in range(num_visualize):
            plt.subplot(1, num_visualize, idx+1)
            # 增加pad参数以防止标题和图片重叠
            imshow(images_vis[idx], title=None)
            plt.title(f"P:{classes[all_preds[idx]]}\nT:{classes[all_labels[idx]]}", fontsize=9, pad=5)
        plt.suptitle("Test Images: Predicted (P) vs True (T)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# 7. main 函数
def main():
    # 获取设备
    device = get_device()
    # 数据预处理与加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载训练集和测试集，只保留猫和狗
    trainset = CatDogCIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CatDogCIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    classes = trainset.classes
    # 可视化训练数据
    visualize_samples(trainloader, classes, num_images=8)
    # 初始化模型、损失函数和优化器
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    print("Start Training...")
    train_model(model, device, trainloader, criterion, optimizer, epochs=5)
    # 测试模型并可视化部分结果
    print("Testing...")
    test_model(model, device, testloader, classes, num_visualize=8)

# 程序入口
if __name__ == "__main__":
    main()