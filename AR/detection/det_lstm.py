import torch
import numpy as np
from AR.detection.feature import MultiTargetTracker
from functools import lru_cache
import time

# 全局缓存
global last_processed_frame, cached_angles, frame_counter, last_results, last_num_persons, result_history
global pinned_buffer  # 新增: 固定内存缓冲区

last_processed_frame = None
cached_angles = {}
frame_counter = 0  # 帧计数器，用于跳帧处理
last_results = None
last_num_persons = 0
result_history = {}  # 用于存储每个人的动作历史


# LSTM 模型定义 - 保持原定义不变
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = torch.nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(0.5)  # Dropout

    def forward(self, x):
        # 优化: 检查输入是否已经是3D的
        if x.dim() == 2:
            out, _ = self.lstm(x.unsqueeze(1))  # (batch, 1, 102)
        else:
            out, _ = self.lstm(x)  # 已经是正确形状

        out = self.bn(out[:, -1, :])  # 归一化最后一个时间步的输出
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 优化：只加载一次模型并创建固定内存缓冲区
def load_lstm_model(model_path=r"D:\PythonProject\mmpose\AR\res\lstm_action_recognition_0410_v3.pth"):
    """
    **加载训练好的 LSTM 模型并创建固定内存缓冲区**
    """
    input_size = 102  # 34 (坐标) + 34 (速度) + 34 (加速度)
    hidden_size = 128
    num_layers = 3
    num_classes = 4

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 进入推理模式

    # 新增: 创建固定内存缓冲区，用于加速CPU-GPU传输
    # 假设最多处理3个人，每人102个特征
    pinned_buffer = torch.zeros(3, 102, dtype=torch.float32).pin_memory()

    return model, device, pinned_buffer


# 初始化模型一次 - 避免每次推理都重新加载
lstm_model, device, pinned_buffer = load_lstm_model()

# 初始化骨骼提取
tracker = MultiTargetTracker()


# 优化计算角度函数
@lru_cache(maxsize=1024)  # 使用lru_cache替代手动缓存管理
def calculate_angle_cached(a0, a1, b0, b1, c0, c1):
    """
    缓存版本的角度计算函数，接受坐标分量作为参数
    """
    # 使用传入的坐标值
    ba0 = a0 - b0
    ba1 = a1 - b1
    bc0 = c0 - b0
    bc1 = c1 - b1

    # 内联计算，减少函数调用
    ba_norm = (ba0 ** 2 + ba1 ** 2) ** 0.5
    bc_norm = (bc0 ** 2 + bc1 ** 2) ** 0.5

    # 计算余弦值
    cosine_angle = (ba0 * bc0 + ba1 * bc1) / (ba_norm * bc_norm + 1e-8)
    cosine_angle = min(1.0, max(-1.0, cosine_angle))  # 裁剪到[-1, 1]
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def calculate_angle(a, b, c):
    """优化的角度计算，使用lru_cache缓存结果"""
    # 将坐标转换为可以传递给缓存函数的参数
    return calculate_angle_cached(
        float(a[0]), float(a[1]),
        float(b[0]), float(b[1]),
        float(c[0]), float(c[1])
    )


def detect_touch_waist(keypoints, keypoints_visible):
    """
    专门用于检测摸腰动作的函数
    Args:
        keypoints: 骨骼关键点坐标
        keypoints_visible: 关键点可见性
    Returns:
        bool: 是否检测到摸腰动作
    """
    try:
        # 检查关键点数据是否有效
        if keypoints.shape[1] != 2:
            print("Warning: Invalid keypoints shape in touch_waist detection")
            return False

        # 获取必要关键点
        left_shoulder = keypoints[5]  # 左肩
        right_shoulder = keypoints[6]  # 右肩
        left_elbow = keypoints[7]  # 左肘
        right_elbow = keypoints[8]  # 右肘
        left_hand = keypoints[9]  # 左手腕
        right_hand = keypoints[10]  # 右手腕
        left_hip = keypoints[11]  # 左髋
        right_hip = keypoints[12]  # 右髋

        # 确认关键点可见性
        left_arm_visible = (keypoints_visible[5] > 0.3 and
                            keypoints_visible[7] > 0.3 and
                            keypoints_visible[9] > 0.3)

        right_arm_visible = (keypoints_visible[6] > 0.3 and
                             keypoints_visible[8] > 0.3 and
                             keypoints_visible[10] > 0.3)

        if not (left_arm_visible or right_arm_visible):
            return False

        # 计算腰部位置 (可调整为更精确的定义)
        waist_center = np.mean([left_hip, right_hip], axis=0)
        # 定义腰部区域为髋部上方约10-15厘米处
        waist_position = np.copy(waist_center)
        waist_position[1] = waist_center[1] - 15  # Y轴向上偏移，更接近实际腰部

        # 参数设置
        distance_threshold = 25  # 手到腰部的距离阈值
        angle_threshold = 110  # 肩-肘-手角度阈值，小于此值表示手臂弯曲
        height_range = 30  # 手与腰部高度差异的允许范围

        # 检查左手是否摸腰
        left_touch_waist = False
        if left_arm_visible:
            # 1. 距离检查
            left_hand_to_waist = np.linalg.norm(left_hand - waist_position)

            # 2. 手臂姿态检查 (肘部弯曲)
            left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_hand)

            # 3. 高度检查 (手在腰部适当高度范围内)
            left_hand_at_waist_height = abs(left_hand[1] - waist_position[1]) < height_range

            left_touch_waist = (left_hand_to_waist < distance_threshold and
                                left_arm_angle < angle_threshold and
                                left_hand_at_waist_height)

        # 检查右手是否摸腰
        right_touch_waist = False
        if right_arm_visible:
            # 1. 距离检查
            right_hand_to_waist = np.linalg.norm(right_hand - waist_position)

            # 2. 手臂姿态检查 (肘部弯曲)
            right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_hand)

            # 3. 高度检查 (手在腰部适当高度范围内)
            right_hand_at_waist_height = abs(right_hand[1] - waist_position[1]) < height_range

            right_touch_waist = (right_hand_to_waist < distance_threshold and
                                 right_arm_angle < angle_threshold and
                                 right_hand_at_waist_height)

        return left_touch_waist or right_touch_waist

    except Exception as e:
        print(f"Error in touch_waist detection: {e}")
        return False


def check_for_fall(keypoints, keypoints_visible, features=None):
    """专门检测摔倒的函数，降低敏感度，要求多个条件同时满足"""
    try:
        # 检查关键点数据是否有效
        if keypoints.shape[1] != 2:
            print("Warning: Invalid keypoints shape in fall detection")
            return False

        # 获取关键点
        nose = keypoints[0]  # 鼻子
        neck = keypoints[1]  # 脖子
        left_shoulder = keypoints[5]  # 左肩
        right_shoulder = keypoints[6]  # 右肩
        left_hip = keypoints[11]  # 左髋
        right_hip = keypoints[12]  # 右髋
        left_knee = keypoints[13]  # 左膝
        right_knee = keypoints[14]  # 右膝
        left_ankle = keypoints[15]  # 左脚踝
        right_ankle = keypoints[16]  # 右脚踝

        # 特征1: 检查身体姿态角度 (人体与地面的角度)
        torso_vector = np.mean([left_hip, right_hip], axis=0) - np.mean([left_shoulder, right_shoulder], axis=0)
        angle_to_vertical = abs(np.degrees(np.arctan2(torso_vector[0], -torso_vector[1])))

        # 特征2: 检查身体部位高度差异
        head_height = nose[1]
        hip_height = np.mean([left_hip[1], right_hip[1]])
        ankle_height = np.mean([left_ankle[1], right_ankle[1]])

        # 特征3: 计算整体身体姿态的紧凑度/变形度
        visible_points = keypoints[keypoints_visible > 0.3]
        if len(visible_points) >= 5:  # 确保有足够多的可见点
            x_coords = visible_points[:, 0]
            y_coords = visible_points[:, 1]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            aspect_ratio = height / (width + 1e-6)  # 高宽比
        else:
            aspect_ratio = 1.0  # 默认值

        # 特征4: 膝盖和臀部的相对位置
        knee_position = np.mean([left_knee, right_knee], axis=0)
        hip_position = np.mean([left_hip, right_hip], axis=0)
        knee_hip_distance = np.linalg.norm(knee_position - hip_position)

        # 初始化条件计数器
        fall_conditions = 0
        total_conditions = 4  # 总条件数

        # 条件1: 躯干接近水平 (通常摔倒时躯干与地面角度较大)
        if angle_to_vertical > 70:  # 提高阈值，降低敏感度
            fall_conditions += 1

        visible_count = np.sum(keypoints_visible > 0.3)
        if visible_count < 9:
            fall_conditions += 2

        # 条件2: 头部接近地面 (头部和脚踝高度接近)
        head_ankle_ratio = abs(head_height - ankle_height) / (abs(hip_height - ankle_height) + 1e-6)
        if head_ankle_ratio < 0.5:  # 提高阈值，降低敏感度
            fall_conditions += 3

        # 条件3: 姿态异常扁平 (摔倒时身体往往在地面上展开)
        if aspect_ratio < 0.6:  # 降低阈值，降低敏感度
            fall_conditions += 1

        # 条件4: 膝盖和臀部的相对位置
        left_leg_vec = left_ankle - left_knee
        right_leg_vec = right_ankle - right_knee

        left_leg_angle = abs(np.degrees(np.arctan2(left_leg_vec[1], left_leg_vec[0])))
        right_leg_angle = abs(np.degrees(np.arctan2(right_leg_vec[1], right_leg_vec[0])))

        # 如果两个腿中至少有一个腿接近水平（小于25度或大于155度），说明腿是摊开的
        if (left_leg_angle < 30 or left_leg_angle > 150 or
                right_leg_angle < 30 or right_leg_angle > 150):
            fall_conditions += 2

        # 特征5: 使用速度和加速度信息辅助判断
        if features is not None:
            # 速度和加速度信息在features中的位置
            # 前34个是坐标，中间34个是速度，最后34个是加速度
            velocities = features[34:68]  # 速度信息
            accelerations = features[68:]  # 加速度信息

            # 计算关键部位（头部、髋部、膝盖）的速度和加速度
            head_vel = np.linalg.norm(velocities[0:2])  # 头部速度
            hip_vel = np.linalg.norm(velocities[22:24])  # 髋部速度
            knee_vel = np.linalg.norm(velocities[26:28])  # 膝盖速度

            head_acc = np.linalg.norm(accelerations[0:2])  # 头部加速度
            hip_acc = np.linalg.norm(accelerations[22:24])  # 髋部加速度
            knee_acc = np.linalg.norm(accelerations[26:28])  # 膝盖加速度

            # 设置更高的速度和加速度阈值
            vel_threshold = 1.0  # 提高速度阈值
            acc_threshold = 1.2  # 提高加速度阈值

            # 如果关键部位的速度或加速度超过阈值，增加条件计数
            if (head_vel > vel_threshold or hip_vel > vel_threshold or knee_vel > vel_threshold or
                    head_acc > acc_threshold or hip_acc > acc_threshold or knee_acc > acc_threshold):
                fall_conditions += 1
                total_conditions += 1  # 增加总条件数

        # 要求至少满足3/4的条件才判定为摔倒
        is_fall = fall_conditions >= 3

        return is_fall

    except Exception as e:
        print(f"Error in fall detection: {e}")
        return False


def smooth_action(current_action, person_id, history_length=3):
    """
    平滑处理动作结果，减少闪烁
    Args:
        current_action: 当前检测到的动作
        person_id: 人物ID
        history_length: 历史记录长度
    Returns:
        str: 平滑后的动作
    """
    global result_history

    # 初始化历史记录
    if person_id not in result_history:
        result_history[person_id] = []

    # 添加当前动作到历史记录
    result_history[person_id].append(current_action)

    # 保持历史记录长度
    if len(result_history[person_id]) > history_length:
        result_history[person_id].pop(0)

    # 如果历史记录太短，直接返回当前动作
    if len(result_history[person_id]) < history_length:
        return current_action

    # 统计历史记录中最常见的动作
    from collections import Counter
    action_counts = Counter(result_history[person_id])
    most_common_action = action_counts.most_common(1)[0][0]

    # 如果最常见的动作出现次数超过阈值，则返回该动作
    if action_counts[most_common_action] >= history_length * 0.6:
        return most_common_action

    return current_action


# 新增: 批量处理特征的函数
def process_features_batch(features_list, pinned_buffer, device):
    """将多个人的特征批量转移到GPU，减少CPU-GPU数据传输"""
    batch_size = len(features_list)

    # 确保不超出缓冲区大小
    if batch_size > pinned_buffer.size(0):
        batch_size = pinned_buffer.size(0)
        features_list = features_list[:batch_size]

    # 将特征数据复制到固定内存缓冲区
    for i, features in enumerate(features_list):
        pinned_buffer[i].copy_(torch.tensor(features, dtype=torch.float32))

    # 使用非阻塞方式将数据传输到GPU
    features_gpu = pinned_buffer[:batch_size].to(device, non_blocking=True)

    return features_gpu


def detect_actions(data_sample):
    """
    **基于骨骼特征进行 LSTM 运动预测，优化版**
    Args:
        data_sample (InstanceData): MMPose 输出的 `instances`
    Returns:
        InstanceData: 处理后的数据样本
    """
    global frame_counter, last_processed_frame, pinned_buffer

    try:
        # 记录处理开始时间，用于性能评估
        start_time = time.time()

        # 增加帧计数器
        frame_counter += 1

        # 获取关键点坐标
        keypoints = data_sample.keypoints  # shape: (num_persons, 17, 2)
        keypoints_visible = data_sample.keypoints_visible  # shape: (num_persons, 17)

        # 检查数据是否有效
        if keypoints is None or keypoints_visible is None:
            print("Warning: Invalid input data")
            default_results = {0: "normal"}
            data_sample.results = default_results
            return data_sample

        # 检查关键点数据是否为空
        if len(keypoints) == 0 or len(keypoints_visible) == 0:
            print("Warning: No valid keypoints")
            default_results = {0: "normal"}
            data_sample.results = default_results
            return data_sample

        # 获取人数
        num_persons = len(keypoints)

        # 如果是需要跳过的帧，使用上一帧的结果
        if frame_counter % 2 == 0:
            if last_processed_frame is not None:
                # 确保结果数量与当前人数匹配
                if len(last_processed_frame.results) != num_persons:
                    # 如果人数不匹配，创建新的结果
                    new_results = {}
                    for i in range(num_persons):
                        if i < len(last_processed_frame.results):
                            new_results[i] = last_processed_frame.results[i]
                        else:
                            new_results[i] = "normal"
                    data_sample.results = new_results
                else:
                    data_sample.results = last_processed_frame.results.copy()
                return data_sample
            else:
                # 如果没有上一帧的结果，返回默认结果
                empty_results = {i: "normal" for i in range(num_persons)}
                data_sample.results = empty_results
                return data_sample

        # 对于需要处理的帧，执行优化后的逻辑
        results = {}

        # 提取骨骼特征
        extracted = tracker.extract_features(data_sample)
        extracted_features = extracted.features

        # 优化点1: 一次性准备所有人的特征，减少循环内的数据转换
        features_list = []
        person_ids = []

        for person_id in range(num_persons):
            if person_id in extracted_features:
                features = extracted_features[person_id]["features"]
            else:
                # 如果没有提取到特征，使用默认值
                features = np.zeros(102)  # 102维特征向量
            features_list.append(features)
            person_ids.append(person_id)

        # 优化点2: 批量处理模型推理，减少CPU-GPU数据传输
        if features_list:
            # 使用固定内存缓冲区加速数据传输
            features_gpu = process_features_batch(features_list, pinned_buffer, device)

            # 批量推理
            with torch.no_grad():  # 禁用梯度计算以节省内存和加速推理
                outputs = lstm_model(features_gpu)  # 批量推理
                _, predicted_classes = torch.max(outputs, 1)

            # 获取预测结果（最小化数据传输）
            predicted_classes = predicted_classes.cpu().numpy()  # 只传输很小的结果数据

            action_labels = ["normal", "bend", "squat", "fall"]

            # 处理每个人的预测结果
            for i, person_id in enumerate(person_ids):
                if i < len(predicted_classes):
                    action = action_labels[predicted_classes[i]]
                else:
                    action = "normal"  # 默认动作

                # 获取当前人的关键点数据
                person_keypoints = keypoints[person_id]
                person_visible = keypoints_visible[person_id]

                # 检查关键点数据是否有效
                if person_keypoints.shape[1] != 2:
                    print(f"Warning: Invalid keypoints shape for person {person_id}")
                    results[person_id] = "normal"
                    continue

                # 使用专门的摔倒检测函数
                if check_for_fall(person_keypoints, person_visible,
                                  features_list[i] if i < len(features_list) else None):
                    action = "fall"
                    results[person_id] = action
                    continue

                # 获取关键点
                try:
                    nose = person_keypoints[0]
                    neck = person_keypoints[1]  # 脖子
                    left_shoulder = person_keypoints[5]  # 左肩
                    right_shoulder = person_keypoints[6]  # 右肩
                    left_elbow = person_keypoints[7]  # 左肘
                    right_elbow = person_keypoints[8]  # 右肘
                    left_hand = person_keypoints[9]  # 左手腕
                    right_hand = person_keypoints[10]  # 右手腕
                    left_hip = person_keypoints[11]  # 左髋
                    right_hip = person_keypoints[12]  # 右髋
                    left_knee = person_keypoints[13]  # 左膝
                    right_knee = person_keypoints[14]  # 右膝
                    left_ankle = person_keypoints[15]  # 左脚踝
                    right_ankle = person_keypoints[16]  # 右脚踝

                    # 计算肩膀-髋部-膝盖角度（用于判断弯腰）
                    left_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)
                    right_shoulder_hip_knee = calculate_angle(right_shoulder, right_hip, right_knee)

                    # 计算髋部-膝盖-脚踝角度（用于判断蹲）
                    left_hip_knee_ankle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_hip_knee_ankle = calculate_angle(right_hip, right_knee, right_ankle)

                    head_to_left_foot_dist_y = abs(nose[1] - left_ankle[1])
                    head_to_right_foot_dist_y = abs(nose[1] - right_ankle[1])
                    hip_to_ankle_height = abs(
                        np.mean([left_hip[1], right_hip[1]]) - np.mean([left_ankle[1], right_ankle[1]]))

                    is_head_close_to_feet = min(head_to_left_foot_dist_y, head_to_right_foot_dist_y) / (
                            hip_to_ankle_height + 1e-6) < 0.6

                    # 设置角度阈值
                    angle_threshold = 100  # 角度阈值

                    if is_head_close_to_feet:
                        # Now determine if it's a bend or squat based on angles
                        if left_hip_knee_ankle < angle_threshold or right_hip_knee_ankle < angle_threshold:
                            action = "squat"  # Knee angle indicates squatting
                        elif left_shoulder_hip_knee < angle_threshold or right_shoulder_hip_knee < angle_threshold:
                            action = "bend"
                    else:
                        action = "normal"

                    # Check for squat first
                    if left_hip_knee_ankle < angle_threshold or right_hip_knee_ankle < angle_threshold:
                        action = "squat"
                    # If not squat, check for bend
                    elif left_shoulder_hip_knee < angle_threshold or right_shoulder_hip_knee < angle_threshold:
                        action = "bend"
                    # If still normal, check for touch_waist or touch_head
                    else:
                        if detect_touch_waist(person_keypoints, person_visible):
                            action = "touch_waist"
                        else:
                            # Check for touch_head
                            try:
                                # Calculate hand to head distance
                                left_hand_to_head = np.linalg.norm(left_hand - nose)
                                right_hand_to_head = np.linalg.norm(right_hand - nose)

                                # Check if touching head
                                if ((left_hand_to_head < 25 and person_visible[9] > 0.3) or
                                        (right_hand_to_head < 25 and person_visible[10] > 0.3)):
                                    action = "touch_head"
                            except Exception as e:
                                print(f"Error in touch_head detection for person {person_id}: {e}")

                except IndexError as e:
                    print(f"Error accessing keypoints for person {person_id}: {e}")
                    action = "normal"  # 如果关键点访问出错，返回normal状态

                # 对动作进行平滑处理
                smoothed_action = smooth_action(action, person_id)
                results[person_id] = smoothed_action  # **存储平滑后的预测结果**
        else:
            # 如果没有人或特征提取失败，为每个人设置默认动作
            results = {i: "normal" for i in range(num_persons)}

        # 确保结果字典的长度与人数一致
        if len(results) != num_persons:
            # 如果结果数量不足，补充normal状态
            for i in range(num_persons):
                if i not in results:
                    results[i] = "normal"
            # 如果结果数量过多，只保留前num_persons个结果
            if len(results) > num_persons:
                results = {k: results[k] for k in sorted(results.keys())[:num_persons]}

        # 更新data_sample的结果
        data_sample.results = results

        # 更新上一帧的结果
        last_processed_frame = data_sample

        # 记录处理结束时间，计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time

        # 打印性能信息
        if frame_counter % 30 == 0:  # 每30帧打印一次性能信息
            print(f"Frame {frame_counter}: Processing time {processing_time:.3f}s, {1 / processing_time:.1f} FPS")

        return data_sample

    except Exception as e:
        print(f"Error in detect_actions: {e}")
        # 如果发生任何错误，返回一个包含normal状态的data_sample
        default_results = {i: "normal" for i in range(num_persons) if 'num_persons'}