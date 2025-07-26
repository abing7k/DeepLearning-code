import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l  # pip install d2l

# 设置真实参数，用于合成数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 生成1000个样本，每个样本是二维特征
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# --------------------------------------------------------
# 构造小批量数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    """封装 features 和 labels 到一个 PyTorch 数据加载器中"""
    dataset = data.TensorDataset(*data_arrays)  # 封装成 dataset
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# --------------------------------------------------------
# 定义模型结构：一个线性层（2维输入 -> 1维输出）
net = nn.Sequential(nn.Linear(2, 1))  # 相当于定义了 wx + b

# 初始化权重参数（均值为0，标准差为0.01），偏置为0
net[0].weight.data.normal_(0, 0.01)  # w初始化
net[0].bias.data.fill_(0)           # b初始化

# --------------------------------------------------------
# 定义损失函数：均方误差 MSE
loss = nn.MSELoss()  # 默认 reduction='mean'

# 定义优化器：使用小批量随机梯度下降，学习率为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# --------------------------------------------------------
# 开始训练
num_epochs = 3  # 训练3轮
for epoch in range(num_epochs):
    for X, y in data_iter:  # 每轮中逐小批处理
        l = loss(net(X), y)  # 前向传播，计算预测损失
        trainer.zero_grad()  # 清空之前的梯度
        l.backward()         # 反向传播，计算梯度
        trainer.step()       # 使用梯度更新参数（SGD）

    # 每轮结束后，计算整个训练集上的平均损失
    train_l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {train_l:f}')

# --------------------------------------------------------
# 查看模型学到的参数，和真实值做比较
w = net[0].weight.data
b = net[0].bias.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)
