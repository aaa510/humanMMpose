from AR.get_labels import merge  # ✅ 直接引入 merge
import numpy as np

# **画面尺寸（用于归一化）**
IMAGE_WIDTH = 1980
IMAGE_HEIGHT = 1080


class MultiTargetTracker:
    def __init__(self):
        """初始化存储变量"""
        self.history = {}  # 存储历史坐标 {person_id: [frame1, frame2, ...]}
        self.center_history = {}  # 存储目标中心点 {person_id: [center1, center2, ...]}

    def extract_features(self, data_sample):
        """
        **从 MMPose data_sample 提取多目标骨骼点**
        - 解析 `merge(data_sample, frame)` 获取 instances
        - **归一化坐标**
        - **计算 速度 & 加速度**
        - **处理目标丢失情况 (用上一帧补全)**
        - **计算 边界框中心点 速度 & 加速度**

        Args:
            data_sample (InstanceData): MMPose 目标检测结果
            frame (np.ndarray): 当前帧图像

        Returns:
            dict: {person_id: {'features': [...], 'center_velocity': (vx, vy), 'center_acceleration': (ax, ay)}}
        """
        instances = merge(data_sample)  # **🔥 获取 instances**
        results = {}

        # **遍历所有目标**
        for i, label_tuple in enumerate(instances.labels):
            person_id = label_tuple[0]  # 'person1' / 'person2'

            # **提取骨骼点**
            keypoints = instances.keypoints[i]  # (17, 2) 的数组
            keypoints = keypoints.flatten()  # 展成 34 维

            # **🔥 归一化坐标**
            keypoints[0::2] /= IMAGE_WIDTH  # **x 坐标归一化**
            keypoints[1::2] /= IMAGE_HEIGHT  # **y 坐标归一化**

            # **存储当前帧数据**
            if person_id not in self.history:
                self.history[person_id] = []
            self.history[person_id].append(keypoints)

            # **补全丢失帧**
            if len(self.history[person_id]) > 3:
                self.history[person_id].pop(0)  # 只存 3 帧历史
            while len(self.history[person_id]) < 3:
                self.history[person_id].insert(0, self.history[person_id][0])

            # **计算速度 & 加速度**
            velocity, acceleration = self._compute_motion_features(self.history[person_id])

            # **计算目标中心点**
            bbox = instances.bboxes[i]
            center_x = (bbox[0] + bbox[2]) / 2 / IMAGE_WIDTH  # **归一化**
            center_y = (bbox[1] + bbox[3]) / 2 / IMAGE_HEIGHT  # **归一化**

            # **存储目标中心点历史**
            if person_id not in self.center_history:
                self.center_history[person_id] = []
            self.center_history[person_id].append((center_x, center_y))

            # **计算中心点速度 & 加速度**
            center_velocity, center_acceleration = self._compute_motion_features(self.center_history[person_id])

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_ratio = (bbox_width * bbox_height) / (IMAGE_WIDTH * IMAGE_HEIGHT)

            # 计算目标中心点坐标


            # **存储最终特征**
            results[person_id] = {
                "features": np.hstack([keypoints, velocity, acceleration]),  # LSTM 输入
                "center_point": [center_x, center_y], # **中心点坐标**
                "center_velocity": center_velocity,  # (vx, vy)
                "center_acceleration": center_acceleration,  # (ax, ay)
                "bbox_ratio": bbox_ratio # 目标框大小占比
            }

        data_sample.features = results

        return data_sample

    def _compute_motion_features(self, history):
        """
        计算运动特征 (速度 & 加速度)

        Args:
            history (list): [frame1, frame2, frame3] (每个 frame 为 np.array)

        Returns:
            tuple: (velocity, acceleration)
        """
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
