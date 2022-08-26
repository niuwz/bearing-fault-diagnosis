import torch
from torch import nn


class CNN(nn.Module):
    '''定义一维卷积神经网络模型'''

    def __init__(self, in_channel=3, out_channel=9):
        super(CNN, self).__init__()
        '''除输入层外，每个层级都包含了卷积、激活和池化三层'''
        '''输出层额外包含了BatchNorm层，提高网络收敛速度以及稳定性'''
        '''第一层卷积核大小为64，之后逐层递减'''
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, padding=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 全连接层定义 引入Dropout机制以提高泛化能力
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channel)
        )
        # 使用softmax函数以计算输出从属于每一类的概率
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        '''前向传播'''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), 1)
        x = self.softmax(x)
        return x
