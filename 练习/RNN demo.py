import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================
# 1. 定义 RNN 模型
# ======================
class RNNClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, num_classes=10):
        super(RNNClassifier, self).__init__()
        # RNN 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层 (从 hidden state 映射到分类结果)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 28, 28) 视为 28 步长，每步输入 28 维
        out, h_n = self.rnn(x)   # out: 所有时间步的输出; h_n: 最后隐藏状态
        # 我们只用最后一步的 hidden state 来分类
        out = self.fc(out[:, -1, :])
        return out


# ======================
# 2. 主程序
# ======================
def main():
    # ----- 设备检测 -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ----- 数据加载 -----
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转成 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准化
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # ----- 模型、损失函数、优化器 -----
    model = RNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ----- 训练 -----
    model.train()
    for epoch in range(1, 3):  # 训练 2 轮
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: (batch, 1, 28, 28) → (batch, 28, 28)
            data = data.squeeze(1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

    # ----- 测试 -----
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.squeeze(1).to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {acc:.2f}%")

# ======================
# 入口
# ======================
if __name__ == "__main__":
    main()
