import torch
from torch import nn


class CNN(nn.Module):
    '''定义一维卷积神经网络模型'''

    def __init__(self, DEVICE, in_channel=3, out_channel=9):
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


class LSTM_CNN(nn.Module):
    '''定义LSTM-CNN网络模型'''

    def __init__(self, DEVICE, in_channel=3, out_channel=9):
        super(LSTM_CNN, self).__init__()
        self.DEVICE = DEVICE
        '''LSTM相关神经元定义'''
        self.lstm_layer1 = nn.LSTM(in_channel, 32)
        self.lstm_layer2 = nn.LSTM(64, 1)
        self.lstm_fc1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        self.lstm_fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

        '''CNN相关神经元定义'''
        self.cnn_layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.cnn_layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel)
        )
        # 使用softmax函数以计算输出从属于每一类的概率
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        '''前向传播'''
        '''*******LSTM*******'''
        # 初始化隐藏神经元
        x_lstm = x.permute(0, 2, 1)
        x_lstm.to(self.DEVICE)
        h1 = torch.zeros(1, 128, 32).to(self.DEVICE)
        c1 = torch.zeros_like(h1).to(self.DEVICE)
        h2 = torch.zeros(1, 128, 1).to(self.DEVICE)
        c2 = torch.zeros_like(h2).to(self.DEVICE)
        y_lstm_ = []
        # 对原时序信号分段
        for i in range(8):
            x_lstm_ = x_lstm[:, i*128:(i+1)*128]
            y, (h1, c1) = self.lstm_layer1(x_lstm_, (h1, c1))
            y = self.lstm_fc1(y)
            y, (h2, c2) = self.lstm_layer2(y, (h2, c2))
            y.to(self.DEVICE)
            y_lstm_.append(y)
        # 合并每一段的结果
        y_lstm = torch.cat(y_lstm_, 1)
        y_lstm = y_lstm.view(y_lstm.size(0), -1)
        y_lstm = self.lstm_fc2(y_lstm)
        '''*******CNN*******'''
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        x = x.view(x.size(0), -1)
        '''******LSTM+CNN******'''
        # 连接LSTM和CNN的输出，并通过全连接神经元
        x = torch.cat([x, y_lstm], 1)
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), 1)
        y = self.softmax(x)
        return y
