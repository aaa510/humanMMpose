import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader

# **设置设备**
device = torch.device("cuda:0")

file_path = r"D:\PythonProject\mmpose\AR\data\merged_datav1.xlsx"

# 读取 Excel 文件
df = pd.read_excel(file_path)
df = df.dropna()

# **提取特征和标签**
X = df.iloc[:, :-1].values  # 前 102 列：坐标、速度、加速度
y = df.iloc[:, -1].values.astype(int)  # 第 103 列：动作标签 (0-5)

# **为多任务学习准备额外标签**
# 这里需要根据你的具体需求调整
# 示例：假设我们需要预测姿态回归值和时间平滑性
y_pose = X[:, :34]  # 使用前34个坐标特征作为姿态回归目标
y_temporal = np.zeros((len(X), 1))  # 时间平滑性标签，这里需要根据实际需求计算

# **划分训练集和测试集（80% 训练，20% 测试）**
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_, _, y_pose_train, y_pose_val = train_test_split(X, y_pose, test_size=0.2, random_state=42, stratify=y)
_, _, y_temporal_train, y_temporal_val = train_test_split(X, y_temporal, test_size=0.2, random_state=42, stratify=y)


# **PyTorch 数据集（多任务版本）**
class MultiTaskMotionDataset(Dataset):
    def __init__(self, X, y_class, y_pose, y_temporal):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)  # 分类标签
        self.y_pose = torch.tensor(y_pose, dtype=torch.float32)  # 姿态回归标签
        self.y_temporal = torch.tensor(y_temporal, dtype=torch.float32)  # 时间平滑性标签

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_pose[idx], self.y_temporal[idx]


# **加载数据**
batch_size = 32
train_dataset = MultiTaskMotionDataset(X_train, y_train, y_pose_train, y_temporal_train)
val_dataset = MultiTaskMotionDataset(X_val, y_val, y_pose_val, y_temporal_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# **定义多任务 LSTM 模型**
class MultiTaskLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, pose_dim, temporal_dim):
        super(MultiTaskLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)  # Batch Normalization

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # 多任务输出头
        self.classification_head = nn.Linear(hidden_size, num_classes)  # 分类头
        self.pose_regression_head = nn.Linear(hidden_size, pose_dim)  # 姿态回归头
        self.temporal_head = nn.Linear(hidden_size, temporal_dim)  # 时间平滑性头

        self.dropout = nn.Dropout(0.5)  # Dropout 0.5

    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # LSTM 需要 3D 输入

        # 计算注意力权重
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 应用注意力权重
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size]

        # 批归一化
        context = self.bn(context)

        # Dropout
        context = self.dropout(context)

        # 多任务输出
        class_out = self.classification_head(context)
        pose_out = self.pose_regression_head(context)
        temporal_out = self.temporal_head(context)

        return class_out, pose_out, temporal_out


# **时间平滑性损失函数**
class TemporalSmoothnessLoss(nn.Module):
    def __init__(self):
        super(TemporalSmoothnessLoss, self).__init__()

    def forward(self, predictions, targets):
        # 计算时间连续性损失
        # 这里是一个简单的实现，你可以根据需求调整
        diff = torch.diff(predictions, dim=0)  # 计算相邻帧的差异
        smoothness_loss = torch.mean(torch.abs(diff))
        return smoothness_loss


# **多任务损失函数**
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # MSE损失权重
        self.gamma = gamma  # 时间平滑性损失权重

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.temporal_loss = TemporalSmoothnessLoss()

    def forward(self, class_pred, pose_pred, temporal_pred, class_target, pose_target, temporal_target):
        # 计算各项损失
        ce_loss = self.ce_loss(class_pred, class_target)
        mse_loss = self.mse_loss(pose_pred, pose_target)
        temp_loss = self.temporal_loss(temporal_pred, temporal_target)

        # 组合损失
        total_loss = self.alpha * ce_loss + self.beta * mse_loss + self.gamma * temp_loss

        return total_loss, ce_loss, mse_loss, temp_loss


# **初始化模型**
input_size = 102  # 34 (坐标) + 34 (速度) + 34 (加速度)
hidden_size = 128
num_layers = 3
num_classes = 4
pose_dim = 34  # 姿态回归维度
temporal_dim = 1  # 时间平滑性维度

model = MultiTaskLSTMModel(input_size, hidden_size, num_layers, num_classes, pose_dim, temporal_dim).to(device)

# **定义损失函数和优化器**
criterion = MultiTaskLoss(alpha=1.0, beta=0.5, gamma=0.3)  # 可调整权重
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# **训练模型**
num_epochs = 100
train_losses, val_losses = [], []
train_ce_losses, train_mse_losses, train_temp_losses = [], [], []

for epoch in range(num_epochs):
    model.train()
    train_loss, train_ce, train_mse, train_temp = 0, 0, 0, 0
    correct, total = 0, 0

    for X_batch, y_class_batch, y_pose_batch, y_temporal_batch in train_loader:
        X_batch = X_batch.to(device)
        y_class_batch = y_class_batch.to(device)
        y_pose_batch = y_pose_batch.to(device)
        y_temporal_batch = y_temporal_batch.to(device)

        optimizer.zero_grad()

        # 前向传播
        class_out, pose_out, temporal_out = model(X_batch)

        # 计算多任务损失
        total_loss, ce_loss, mse_loss, temp_loss = criterion(
            class_out, pose_out, temporal_out,
            y_class_batch, y_pose_batch, y_temporal_batch
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 记录损失
        train_loss += total_loss.item()
        train_ce += ce_loss.item()
        train_mse += mse_loss.item()
        train_temp += temp_loss.item()

        # 计算分类准确率
        _, predicted = torch.max(class_out, 1)
        total += y_class_batch.size(0)
        correct += (predicted == y_class_batch).sum().item()

    # 记录训练损失
    train_losses.append(train_loss / len(train_loader))
    train_ce_losses.append(train_ce / len(train_loader))
    train_mse_losses.append(train_mse / len(train_loader))
    train_temp_losses.append(train_temp / len(train_loader))
    train_acc = correct / total

    # **验证阶段**
    model.eval()
    val_loss, val_ce, val_mse, val_temp = 0, 0, 0, 0
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_class_batch, y_pose_batch, y_temporal_batch in val_loader:
            X_batch = X_batch.to(device)
            y_class_batch = y_class_batch.to(device)
            y_pose_batch = y_pose_batch.to(device)
            y_temporal_batch = y_temporal_batch.to(device)

            class_out, pose_out, temporal_out = model(X_batch)
            total_loss, ce_loss, mse_loss, temp_loss = criterion(
                class_out, pose_out, temporal_out,
                y_class_batch, y_pose_batch, y_temporal_batch
            )

            val_loss += total_loss.item()
            val_ce += ce_loss.item()
            val_mse += mse_loss.item()
            val_temp += temp_loss.item()

            _, predicted = torch.max(class_out, 1)
            total += y_class_batch.size(0)
            correct += (predicted == y_class_batch).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_acc = correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"  Train - Total: {train_losses[-1]:.4f}, CE: {train_ce_losses[-1]:.4f}, "
          f"MSE: {train_mse_losses[-1]:.4f}, Temp: {train_temp_losses[-1]:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   - Total: {val_losses[-1]:.4f}, Acc: {val_acc:.4f}")

    scheduler.step()  # 更新学习率

# **绘制损失曲线**
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 总损失
axes[0, 0].plot(train_losses, label="Train Total Loss")
axes[0, 0].plot(val_losses, label="Val Total Loss")
axes[0, 0].set_title("Total Loss")
axes[0, 0].legend()

# 分类损失
axes[0, 1].plot(train_ce_losses, label="Train CE Loss")
axes[0, 1].set_title("CrossEntropy Loss")
axes[0, 1].legend()

# MSE损失
axes[1, 0].plot(train_mse_losses, label="Train MSE Loss")
axes[1, 0].set_title("MSE Loss")
axes[1, 0].legend()

# 时间平滑性损失
axes[1, 1].plot(train_temp_losses, label="Train Temporal Loss")
axes[1, 1].set_title("Temporal Smoothness Loss")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# **生成混淆矩阵**
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_class_batch, y_pose_batch, y_temporal_batch in val_loader:
        X_batch = X_batch.to(device)
        y_class_batch = y_class_batch.to(device)
        y_pose_batch = y_pose_batch.to(device)
        y_temporal_batch = y_temporal_batch.to(device)

        class_out, _, _ = model(X_batch)  # 只需要分类输出
        _, predicted = torch.max(class_out, 1)
        y_true.extend(y_class_batch.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# **计算混淆矩阵**
conf_matrix = confusion_matrix(y_true, y_pred, normalize="true") * 100  # 计算百分比

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["normal", "bend", "squat", "fall"],
            yticklabels=["normal", "bend", "squat", "fall"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (%)")
plt.show()

# **打印分类报告**
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "bend", "squat", "fall"]))

# **保存模型**
model_path = "./res/multitask_lstm_action_recognition_523_v2.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ 训练完成，多任务模型权重已保存至 {model_path}")

# **打印最终损失组合信息**
print("\n=== Multi-task Loss Combination ===")
print(f"α (CrossEntropy weight): {criterion.alpha}")
print(f"β (MSE weight): {criterion.beta}")
print(f"γ (Temporal Smoothness weight): {criterion.gamma}")
print("Loss_total = α * CrossEntropyLoss + β * MSE_Loss + γ * TemporalSmoothnessLoss")