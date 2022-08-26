from scipy import io
import numpy as np
import torch
import pandas as pd
from fault_diag_utils import *

if __name__ == "__main__":
    # 读取测试数据
    data = io.loadmat("轴承故障诊断/data/Fault_Diag_Data_for_student.mat")
    test_data = torch.tensor(
        data["TestDataArray"], dtype=torch.float32).permute(2, 1, 0)
    # 获取训练设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载已训练好的模型
    net = torch.load("轴承故障诊断/cnn_net.pth").to(DEVICE)
    print(f"DEVICE:{DEVICE}")
    net.eval()
    predict = []
    for x in test_data:
        x = x.reshape(1, x.shape[0], x.shape[1]).to(DEVICE)
        # 计算预测值
        y = net(x)
        # 获取预测标签
        output = y.max(1, keepdim=True)[1]
        predict.append(get_label(output.item()))
    # 将测试数据及标签转换为pd.DataFrame
    df = {
        "index": [i for i in range(len(test_data))],
        "label": predict
    }
    df = pd.DataFrame(df)
    # 保存测试集标签
    df.to_excel("轴承故障诊断/results.xlsx", index=False)
