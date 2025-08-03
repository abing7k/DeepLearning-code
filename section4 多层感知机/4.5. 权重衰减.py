import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 全局参数
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

# 生成数据
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, w.shape[0]))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_data():
    X_train, y_train = synthetic_data(true_w, true_b, n_train)
    X_test, y_test = synthetic_data(true_w, true_b, n_test)
    batch_size = 5
    train_iter = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), batch_size, shuffle=False)
    return train_iter, test_iter

# 主训练函数（简洁实现）
def train_concise(wd, train_iter, test_iter):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # 使用两个参数组：一个加 weight decay，一个不加
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}
    ], lr=lr)

    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (
                d2l.evaluate_loss(net, train_iter, loss),
                d2l.evaluate_loss(net, test_iter, loss)
            ))

    print('w的L2范数：', net[0].weight.norm().item())
    plt.show()

# main函数
def main():
    train_iter, test_iter = load_data()
    print("不使用权重衰减：")
    train_concise(0, train_iter, test_iter)
    print("\n使用权重衰减（L2正则）wd=3：")
    train_concise(3, train_iter, test_iter)
    print("\n使用权重衰减（L2正则）wd=10：")
    train_concise(10, train_iter, test_iter)
if __name__ == '__main__':
    main()
