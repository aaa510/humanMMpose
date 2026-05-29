import numpy as np
import math

# **权重与计算参数**
K_VALUES = {
    "speed": 0.2303, "acceleration": 0.2878,
    "score": 0.0461, "score_rate": 0.4605,
    "lose_score": 0.4000, "lose_rate": 0.4605
}

WEIGHTS = {
    "attack": {"score_rate": 0.4574, "score": 0.2498, "action": 0.1347, "position": 0.0804, "acceleration": 0.0489,
               "speed": 0.0288},
    "defense": {"lose_rate": 0.6333, "lose_score": 0.2605, "action": 0.1062},
    "decision": {"score_rate": 0.6333, "score": 0.2605, "action": 0.1062},
    "agility": {"action": 0.5714, "acceleration": 0.2857, "speed": 0.1429},
    "final": {"attack": 0.5579, "defense": 0.2633, "decision": 0.1219, "agility": 0.0569}
}

# **动作评分表**
ACTION_SCORES = {
    "attack": {"Stand": 0.2, "Run": 0.4, "Jump": 0.6, "Throw": 0.8},
    "defense": {"Walk": 0.2, "Run": 0.5, "Squat": 0.8},
    "decision": {"Jump": 0.6, "Throw": 0.9},
    "agility": {"Run": 0.6, "Jump": 0.8}
}

Action_Labels = ["Stand", "Walk", "Run", "Jump", "Throw", "Squat"]

class ThreatAssessment:
    def __init__(self):
        self.frame_count = 0
        self.scores = {}  # {person_id: {"score": x, "lose_score": y}}
        self.global_stats = {}  # 维护全局统计数据

    def update_scores(self, person_ids):
        """模拟UDP数据，每30帧更新一次"""
        self.frame_count += 1

        # ✅ 确保 person_ids 可迭代
        person_ids = list(person_ids)
        for person_id in person_ids:
            if person_id not in self.scores:
                self.scores[person_id] = {"score": 0, "lose_score": 0}  # **首次初始化**

        if self.frame_count % 30 == 0:
            for person_id in person_ids:
                if isinstance(person_id, str):  # 确保 person_id 是字符串
                    self.scores[person_id]["score"] += 1  # ✅ **增加击打得分**
                    self.scores[person_id]["lose_score"] += 1  # ✅ **增加失分**
                else:
                    print(f"⚠️ 无效的 person_id: {person_id}, 类型: {type(person_id)}")  # 调试信息

        return {
            person_id: {
                "score": self.scores[person_id]["score"],
                "lose_score": self.scores[person_id]["lose_score"],
                "score_rate": 1,  # 变化率始终为 1
                "lose_rate": 1
            }
            for person_id in self.scores  # ✅ **确保所有已记录的人都有返回值**
        }

    def update_global_stats(self, data_sample, person_id, speed, acceleration, action, score, lose_score, score_rate,
                            lose_rate, threat_index):
        """将全局统计数据存入 `data_sample` 中"""

        # **✅ 初始化 global_stats（仅在第一次访问时）**
        if not hasattr(data_sample, "global_stats"):
            data_sample.set_field({}, "global_stats")  # 采用 set_field 方式

        # **✅ 确保该 person_id 也有初始化**
        if person_id not in data_sample.global_stats:
            data_sample.global_stats[person_id] = {
                "max_speed": 0, "min_speed": float("inf"),
                "max_acceleration": 0, "min_acceleration": float("inf"),
                "action_count": {key: 0 for key in Action_Labels},
                "total_score": 0, "total_lose_score": 0,
                "score_rate": 0, "lose_rate": 0,
                "threat_history": []
            }

        stats = data_sample.global_stats[person_id]

        # **✅ 更新最大/最小速度 & 加速度**
        stats["max_speed"] = max(stats["max_speed"], speed)
        stats["min_speed"] = min(stats["min_speed"], speed)
        stats["max_acceleration"] = max(stats["max_acceleration"], acceleration)
        stats["min_acceleration"] = min(stats["min_acceleration"], acceleration)

        # **✅ 更新动作统计**
        if action in stats["action_count"]:
            stats["action_count"][action] += 1
        else:
            stats["action_count"][action] = 1  # 以防 action 不在字典里

        # **✅ 更新得分统计**
        stats["total_score"] = score  # 逐帧累加总得分
        stats["total_lose_score"] = lose_score  # 逐帧累加总失分

        # **✅ 计算得分变化率**
        stats["score_rate"] = score_rate  # 直接写入最新的得分变化率
        stats["lose_rate"] = lose_rate  # 直接写入最新的失分变化率

        # **✅ 记录威胁指数历史**
        stats["threat_history"].append(threat_index)

        # **✅ 确保 `data_sample.global_stats` 被正确写回**
        data_sample.set_field(data_sample.global_stats, "global_stats")


# **计算归一化得分**
def normalize_value(value, k, positive=True):
    """ 归一化公式 y = 1 - e^(-kx) (正相关) / y = e^(-kx) (负相关) """
    if positive:
        return 1 - math.exp(-k * value)
    return math.exp(-k * value)


# **处理数据**
def process_features(data_sample, threat_model):
    """
    读取 `extract.py` + `det_lstm.py` 结果，归一化 & 计算威胁指数
    Args:
        data_sample: MMPose 提取的人体特征
        frame: 当前帧图像
        threat_model: 威胁评分模型 (负责计分模拟)
    Returns:
        dict: {'person_id': {'attack': x, 'defense': y, 'decision': z, 'agility': w, 'threat': v}}
    """
    # tracker = MultiTargetTracker()
    # motion_data = tracker.extract_features(data_sample)  # 提取 运动属性
    motion_data = data_sample.features
    # action_data = detect_actions(data_sample)  # 提取 动作识别
    action_data = data_sample.results
    score_info = threat_model.update_scores(motion_data.keys())  # **获取最新的计分数据**

    results = {}

    for person_id in motion_data.keys():
        # **基础运动数据**
        motion = motion_data[person_id]
        bbox_ratio = motion["bbox_ratio"]

        # **计算速度 & 加速度模长**
        speed = np.linalg.norm(motion["center_velocity"])  # 🚀 计算速度模长
        acceleration = np.linalg.norm(motion["center_acceleration"])  # 🚀 计算加速度模长

        # **归一化**
        norm_speed = normalize_value(speed, K_VALUES["speed"], positive=True)
        norm_acceleration = normalize_value(acceleration, K_VALUES["acceleration"], positive=True)

        # **获取实时计分**
        score = score_info[person_id]["score"]
        lose_score = score_info[person_id]["lose_score"]
        score_rate = score_info[person_id]["score_rate"]
        lose_rate = score_info[person_id]["lose_rate"]

        norm_score = normalize_value(score, K_VALUES["score"], positive=True)
        norm_lose_score = normalize_value(lose_score, K_VALUES["lose_score"], positive=False)
        norm_score_rate = normalize_value(score_rate, K_VALUES["score_rate"], positive=True)
        norm_lose_rate = normalize_value(lose_rate, K_VALUES["lose_rate"], positive=False)

        # **动作识别**
        action = action_data.get(person_id, "Stand")

        # **计算四个二级指标**
        attack = (
            WEIGHTS["attack"]["score_rate"] * norm_score_rate +
            WEIGHTS["attack"]["score"] * norm_score +
            WEIGHTS["attack"]["action"] * ACTION_SCORES["attack"].get(action, 0) +
            WEIGHTS["attack"]["position"] * bbox_ratio +
            WEIGHTS["attack"]["acceleration"] * norm_acceleration +
            WEIGHTS["attack"]["speed"] * norm_speed
        )

        defense = WEIGHTS["defense"]["lose_rate"] * norm_lose_rate + WEIGHTS["defense"]["lose_score"] * norm_lose_score + WEIGHTS["defense"]["action"] * ACTION_SCORES["defense"].get(action, 0)

        decision = WEIGHTS["decision"]["score_rate"] * norm_score_rate + WEIGHTS["decision"]["score"] * norm_score + WEIGHTS["decision"]["action"] * ACTION_SCORES["decision"].get(action, 0)

        agility = WEIGHTS["agility"]["action"] * ACTION_SCORES["agility"].get(action, 0) + WEIGHTS["agility"]["acceleration"] * norm_acceleration + WEIGHTS["agility"]["speed"] * norm_speed

        threat = attack * WEIGHTS["final"]["attack"] + defense * WEIGHTS["final"]["defense"] + decision * WEIGHTS["final"][
                "decision"] + agility * WEIGHTS["final"]["agility"]

        results[person_id] = {"attack": attack, "defense": defense, "decision": decision, "agility": agility, "threat": threat}

        if not hasattr(data_sample, "static"):
            data_sample.set_field({}, "static")  # **初始化 `static`**

        data_sample.static[person_id] = {
            "speed": speed, "acceleration": acceleration, "action": action,
            "score": score, "lose_score": lose_score,
            "score_rate": score_rate, "lose_rate": lose_rate,
            "threat": threat
        }
    data_sample.dy_threat = results
    return data_sample
