import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义模型结构
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        scores = self.attention_weights(lstm_output)  # [batch_size, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch_size, seq_len]
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]

        # context vector as weighted sum of lstm_output
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

class ResNet_CNN_LSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, num_layers=1, dropout=0.5):
        super(ResNet_CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(
            input_size=64,  # 调整后的输入通道
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.attention = Attention(hidden_size)

        # 新增的全连接层
        self.fc_extra = nn.Linear(hidden_size, 128)  # 额外的全连接层

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)  # 调整后的全连接层
        self.fc2 = nn.Linear(64, 7)  # 输出层

    def forward(self, x):
        # CNN部分
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.permute(0, 2, 1)  # 调整维度以适应LSTM输入

        # LSTM部分
        lstm_out, _ = self.lstm(x)

        # Attention机制
        context_vector, attention_weights = self.attention(lstm_out)

        # 通过新增的全连接层
        x = self.fc_extra(context_vector)
        x = F.relu(x)
        x = self.dropout(x)  # 在此添加Dropout层

        # 通过原来的全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # 再次添加Dropout层
        x = self.fc2(x)

        return x

# 加载保存的模型
model_path = r"./model/resnet_lstm_cnn_final_model.pth"  # 修改为实际模型路径
net = ResNet_CNN_LSTM(input_size=256, hidden_size=128, num_layers=1, dropout=0.5)
net.load_state_dict(torch.load(model_path))
net.eval()  # 设置模型为评估模式

# 加载新数据
new_data_path = r"\新数据.npy"  # 修改为实际数据路径
new_labels_path = r"新标签.npy"  # 修改为实际标签路径
new_data = np.load(new_data_path)
new_labels = np.load(new_labels_path)

# 数据转换为PyTorch张量
new_data_tensor = torch.tensor(new_data, dtype=torch.float32).unsqueeze(1)  # 添加一个通道维度
new_labels_tensor = torch.tensor(new_labels, dtype=torch.long)

# 预测并计算准确率
with torch.no_grad():
    outputs = net(new_data_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == new_labels_tensor).sum().item()
    total = new_labels_tensor.size(0)
    accuracy = 100 * correct / total

print(f"Prediction Accuracy: {accuracy:.2f}%")
