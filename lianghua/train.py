from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

#以下为下载数据集
def train(model, device, train_loader, optimizer, epoch):   #训练集
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()    #交叉熵损失函数
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()       #梯度归0
        output = model(data)
        loss = lossLayer(output, target)    #损失层（CNN预测值，真实标签）
        loss.backward()    #反向传播计算得到每个参数的梯度值
        optimizer.step()   #通过梯度下降执行一步参数更新

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def test(model, device, test_loader):   #测试集
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        #求准确率
        pred = output.argmax(dim=1, keepdim=True)  #返回指定维度最大值序号
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    batch_size = 64    #单次传递参数数据
    test_batch_size = 64  #
    seed = 1    #产生的随机数固定
    epochs = 15   #一个等于使用训练集中全部数据训练一次
    lr = 0.01   #学习率
    momentum = 0.5   #动量（优化算法）
    save_model = True  #生成冻结图文件
    using_bn = True  #样本标准化

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    #本机使用CPU

#数据读取
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),    #转换为张量
                           transforms.Normalize((0.1307,), (0.3081,))    #标准化
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    if using_bn:     #批量归一化折叠
        model = NetBN().to(device)
    else:
        model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)   #SGD优化器

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:       #保存模型
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        if using_bn:
            torch.save(model.state_dict(), 'ckpt/mnist_cnnbn.pt')
        else:
            torch.save(model.state_dict(), 'ckpt/mnist_cnn.pt')