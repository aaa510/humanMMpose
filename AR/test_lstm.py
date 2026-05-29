import torch
import torch.nn as nn
import numpy as np


# **LSTM 模型定义（必须和训练时一致）**
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)  # Batch Normalization (必须保持)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout (必须保持)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))  # LSTM 需要 3D 输入
        out = self.bn(out[:, -1, :])  # 归一化最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)
        return out


# **加载模型**
def load_lstm_model(model_path=r"C:\PythonProject\mmpose\mmpose\AR\res\lstm_action_recognition_test01.pth"):
    input_size = 34  # 34 (坐标) + 34 (速度) + 34 (加速度)
    hidden_size = 128
    num_layers = 3
    num_classes = 6

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

    # **加载权重**
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


# **计算速度 & 加速度**
def compute_motion_features(history):
    """计算速度（dx/dt）和加速度（dv/dt），单位时间间隔 1/30s"""
    history = np.array(history)
    if len(history) < 2:
        velocity = np.zeros_like(history[0])
    else:
        velocity = (history[-1] - history[-2]) * 30  # 速度 = Δx / Δt

    if len(history) < 3:
        acceleration = np.zeros_like(velocity)
    else:
        acceleration = (velocity - ((history[-2] - history[-3]) * 30)) * 30  # 加速度 = Δv / Δt

    return velocity, acceleration


# **LSTM 推理**
def predict_action(model, keypoints_history):
    """使用 LSTM 预测当前动作"""
    velocity, acceleration = compute_motion_features(keypoints_history)

    # **拼接特征**
    lstm_input = np.hstack([keypoints_history[-1], velocity, acceleration])  # 坐标 + 速度 + 加速度
    lstm_input = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0)

    # **模型预测**
    with torch.no_grad():
        output = model(lstm_input)
        _, predicted_class = torch.max(output, 1)

    action_labels = ["Stand", "Walk", "Run", "Jump", "Throw", "Squat"]
    return action_labels[predicted_class.item()]


# **测试 LSTM**
if __name__ == "__main__":
    model = load_lstm_model()

    # **模拟关键点输入（34 维坐标）**
    dummy_keypoints = np.random.rand(34)  # 归一化坐标
    keypoints_history = [dummy_keypoints] * 3  # 假设存储 3 帧历史

    action = predict_action(model, keypoints_history)
    print(f"🎯 预测动作: {action}")
