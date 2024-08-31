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
# 定义从文件夹加载数据的函数
def load_data_from_folder(folder_path):
    data = np.load(os.path.join(folder_path, "train_data.npy"))
    labels = np.load(os.path.join(folder_path, "train_labels.npy"))
    return data, labels

base_path = r"./data"

train_data_path = os.path.join(base_path, "train_data.npy")
train_labels_path = os.path.join(base_path, "train_labels.npy")
val_data_path = os.path.join(base_path, "val_data.npy")
val_labels_path = os.path.join(base_path, "val_labels.npy")
test_data_path = os.path.join(base_path, "test_data.npy")
test_labels_path = os.path.join(base_path, "test_labels.npy")

train_data = np.load(train_data_path)
train_labels = np.load(train_labels_path)
val_data = np.load(val_data_path)
val_labels = np.load(val_labels_path)
test_data = np.load(test_data_path)
test_labels = np.load(test_labels_path)


# 将数据转换为PyTorch张量
train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # 添加一个通道维度
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
val_data_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
test_data_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
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

# 实例化模型
net = ResNet_CNN_LSTM(input_size=256, hidden_size=128, num_layers=1, dropout=0.5)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 定义RAdam优化器
import torch.optim as optim

# 定义AdamW优化器
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-3)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)





# 记录损失和准确率
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# 检查点保存路径
checkpoint_path = "resnet_cnn_lstm_checkpoint.pth"

# 尝试恢复检查点
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_accuracies = checkpoint['val_accuracies']
    print(f"Checkpoint loaded, resuming from epoch {start_epoch}")

# 训练模型
# 训练模型
num_epochs = 100
for epoch in range(start_epoch, num_epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    net.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # 使用验证集损失更新学习率
    scheduler.step(val_loss)


    # 每10轮保存一次检查点
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")


# 测试集准确率
net.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 绘制训练过程中的损失曲线和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plot_save_path = "Resnet_CNN_LSTM-training_plot.png"
plt.savefig(plot_save_path)
plt.show()

# 保存训练好的模型
model_save_path = "resnet_lstm_cnn_final_model.pth"
torch.save(net.state_dict(), model_save_path)
print(f"Trained model saved to {model_save_path}")

print(f"Training plot saved to {plot_save_path}")
print("Finished Training")
