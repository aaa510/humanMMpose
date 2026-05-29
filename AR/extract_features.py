import pandas as pd
import numpy as np
import os

# 输入和输出文件夹
input_folder = r"D:\PythonProject\mmpose\AR\data\experiment_data"
output_folder = r"D:\PythonProject\mmpose\AR\data\experiment_after"

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 帧率和时间间隔
fps = 30
dt = 1 / fps

# 归一化函数
def normalize_coordinates(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        if col % 2 == 0:  # X 坐标
            df_norm[col] = df_norm[col] / 640
        else:  # Y 坐标
            df_norm[col] = df_norm[col] / 480
    return df_norm.round(4)

# 计算速度
def compute_velocity(df_norm):
    velocity_df = df_norm.diff().fillna(0) / dt
    return velocity_df.round(4)

# 计算加速度
def compute_acceleration(velocity_df):
    acceleration_df = velocity_df.diff().fillna(0) / dt
    return acceleration_df.round(4)

# 遍历处理所有 Excel 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_excel(file_path)

        # 列名强制为整数
        df.columns = df.columns.astype(int)

        # ✅ 去掉34列之后的无用数据，只保留前34列
        df = df.iloc[:, :34]

        # 步骤1：归一化
        df_normalized = normalize_coordinates(df)

        # 步骤2：速度计算
        df_velocity = compute_velocity(df_normalized)

        # 步骤3：加速度计算
        df_acceleration = compute_acceleration(df_velocity)

        # 步骤4：重命名列
        df_velocity.columns = [f'V_{i}' for i in range(34)]
        df_acceleration.columns = [f'A_{i}' for i in range(34)]

        # 步骤5：拼接列（写死顺序）
        df_normalized.columns = list(range(34))  # 保证坐标列是0~33
        df_final = pd.concat([df_normalized, df_velocity, df_acceleration], axis=1)

        # 步骤6：保存
        output_path = os.path.join(output_folder, file_name)
        df_final.to_excel(output_path, index=False)

        print(f"✅ {file_name} 处理完成，已保存至 {output_path}")

print("🎉 所有文件处理完成！")
