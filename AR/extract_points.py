import os
import numpy as np
import pandas as pd
import time

# Excel 文件路径
OUTPUT_EXCEL = r'D:\PythonProject\mmpose\AR\data\experiment_data\normal.xlsx'


def extract_keypoints(data_samples, prev_frame_data, max_columns):
    """
    提取当前帧的关键点数据，并用上一帧数据填补缺失的部分。

    :param data_samples: 当前帧检测到的所有人物的 keypoints 数据（列表）
    :param prev_frame_data: 上一帧的 keypoints 数据（用于填充缺失人物）
    :param max_columns: 目前整个视频中最多的人数列数（固定列数）
    :return: 当前帧的关键点数据，更新后的最大列数
    """
    frame_keypoints = []  # 记录当前帧的所有人数据

    for frame_data in data_samples:  # 遍历每帧的数据
        if not isinstance(frame_data, list) or len(frame_data) == 0:
            continue  # 如果 frame_data 不是列表或为空，跳过

        for person_data in frame_data:  # 遍历当前帧的每个人
            keypoints = person_data.get('keypoints', None)  # 直接从字典获取 keypoints
            if keypoints is None or len(keypoints) == 0:
                continue  # 没有关键点，跳过

            # 关键点数据是二维列表 [[x0, y0], [x1, y1], ..., [x16, y16]]
            keypoints_flattened = np.array(keypoints).flatten()  # 转换为一维数组 (34,)
            frame_keypoints.extend(keypoints_flattened)  # 添加到当前帧

    # **更新最大列数**（取当前帧数据列数与历史最大列数中的较大值）
    max_columns = max(max_columns, len(frame_keypoints))

    # **填补当前帧数据**（如果人数比上一帧少，用上一帧数据补齐）
    if prev_frame_data is not None and len(frame_keypoints) < max_columns:
        missing_data = prev_frame_data[len(frame_keypoints):]  # 取上一帧的剩余数据
        frame_keypoints.extend(missing_data)

    # **更新上一帧数据**
    prev_frame_data = frame_keypoints.copy()

    return frame_keypoints, prev_frame_data, max_columns  # 返回当前帧数据和更新后的最大列数


def save_to_excel(keypoint_list, max_columns):
    """
    将关键点数据存储到 Excel，并确保所有行的列数一致。
    """
    if not keypoint_list:
        print("未检测到任何关键点数据，Excel 文件未生成。")
        return

    # **统一所有帧的数据列数**（用 NaN 填充不足的部分）
    for i in range(len(keypoint_list)):
        if len(keypoint_list[i]) < max_columns:
            keypoint_list[i].extend([np.nan] * (max_columns - len(keypoint_list[i])))

    # **自动生成列名**
    num_people = max_columns // 34  # 计算最大人数
    # columns = [f'x{p}_{j}' for p in range(num_people) for j in range(17)] + \
    #           [f'y{p}_{j}' for p in range(num_people) for j in range(17)]

    # **创建 DataFrame 并保存到 Excel**
    # df = pd.DataFrame(keypoint_list, columns=columns)
    df = pd.DataFrame(keypoint_list)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f'✅ 关键点数据已保存至 {OUTPUT_EXCEL}')
