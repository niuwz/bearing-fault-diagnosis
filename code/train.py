import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, dataset, random_split
from matplotlib import pyplot as plt
import seaborn as sns
from fault_diag_utils import *
from cnn_model import CNN as modle
sns.set()


def train(net: modle, dataloader: DataLoader, loss_func, optimizer: optim.Adam, epoch: int):
    '''训练CNN模型'''
    global DEVICE, LOSSES
    # 训练模式
    net.train()
    loss_mean = 0
    for i, (x, label) in enumerate(dataloader):
        x, label = x.to(DEVICE), label.to(DEVICE)
        # 清空梯度
        optimizer.zero_grad()
        # 计算预测值
        y = net(x).to(DEVICE)
        # 计算损失
        loss = loss_func(y, label)
        # 误差反向传播与参数更新
        loss.backward()
        optimizer.step()
        # 统计平均损失
        loss_mean += loss.item()
        # 打印训练及损失信息
        if ((epoch+1) % 10 == 0) and ((i+1) % 8 == 0):
            print("Train Epoch: {}\t[{: >4}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch+1, i*len(x), len(dataloader.dataset), 100*i/len(dataloader), loss.item()))
    LOSSES.append(np.mean(loss_mean))


def test(net: modle, dataloader: DataLoader, loss_func, datatype: str):
    '''计算模型准确率'''
    global DEVICE, ACCS
    # 测试模式
    net.eval()
    test_loss = 0
    cnt = 0
    with torch.no_grad():
        for x, label in dataloader:
            x, label = x.to(DEVICE), label.to(DEVICE)
            # 计算预测值
            y = net(x).to(DEVICE)
            test_loss += loss_func(y, label)
            # 根据预测概率获取预测类别
            predict = y.max(1, keepdim=True)[1]
            cnt += predict.eq(label.view_as(predict)).sum().item()
    # 计算损失和正确率并打印
    test_loss /= len(dataloader.dataset)
    acc = 100*cnt/len(dataloader.dataset)
    ACCS.append(acc)
    print("{} Data:  Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(datatype,
                                                                             test_loss, cnt, len(dataloader.dataset), acc))


if __name__ == "__main__":
    '''定义相关超参数'''
    global DEVICE, BATCH_SIZE, EPOCHS, LENGTH, PATH, LOSSES, ACCS
    # 样本长度
    LENGTH = 1024
    # 批尺寸
    BATCH_SIZE = 64
    # 训练迭代次数
    EPOCHS = 50
    # 重叠采样的偏移量
    STRIDE = 128
    PATH = "轴承故障诊断/data"
    # 定义训练设备这里使用GPU进行加速
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保存损失以及准确率信息
    LOSSES = []
    ACCS = []
    print(f"DEVICE:{DEVICE}")
    # 读取数据
    files = os.listdir(PATH)
    label = []
    data = []
    for file in files:
        if "csv" not in file:
            continue
        labeli, xi = file_read(PATH + "/" + file)
        # 重叠采样
        for j in range(0, len(xi)-LENGTH, STRIDE):
            label.append(labeli)
            data.append(xi[j:j+LENGTH, :].T)
        # 额外截取最后一段数据
        label.append(labeli)
        data.append(xi[-LENGTH:, :].T)
    # 定义数据集
    ds = Data_set(data, label)
    '''数据集分割'''
    train_size = int(0.7*len(ds))
    test_size = len(ds) - train_size
    train_loader, test_loader = random_split(ds, [train_size, test_size])
    train_loader = DataLoader(train_loader, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_loader, BATCH_SIZE, shuffle=True)
    '''定义网络模型、优化器和损失函数'''
    net = modle(3, 9).to(DEVICE)
    optimizer = optim.Adam(net.parameters())
    # 损失函数使用交叉熵函数
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train(net, train_loader, loss_func, optimizer, epoch)
        if (epoch+1) % 25 == 0:
            print()
            test(net,  train_loader, loss_func, "训练")
            test(net, test_loader, loss_func, "验证")
            print()
    # 保存模型
    torch.save(net, "轴承故障诊断/cnn_net.pth")
    '''绘图部分'''
    # plt.plot(LOSSES)
    # plt.xlabel("Epochs")
    # plt.xticks(np.arange(0, EPOCHS+1, 1))
    # plt.ylabel("Loss")

    # plt.figure()
    # plt.plot(ACCS)
    # plt.xlabel("Epochs")
    # plt.xticks(np.arange(0, EPOCHS+1, 1))
    # plt.ylabel("Accuracy%")
    # plt.show()
