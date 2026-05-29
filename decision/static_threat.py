import numpy as np
import math
from decision.dynamic_threat import ThreatAssessment, normalize_value

# **权重与计算参数**
K_VALUES = {
    "max_speed": 0.2303, "max_acceleration": 0.2308,
    "min_speed": 0.2303, "min_acceleration": 0.2308,
    "score_rate": 0.4605, "lose_rate": 0.4605,
    "stand_frames": 0.0001, "run_frames": 0.0001,
    "jump_frames": 0.0001, "throw_frames": 0.0001,
    "walk_frames": 0.0001, "squat_frames": 0.0001
}

WEIGHTS = {
    "attack": {
        "max_speed": 0.2805, "max_acceleration": 0.4504,
        "stand_frames": 0.0308, "run_frames": 0.2510,
        "jump_frames": 0.0857, "throw_frames": 0.1456
    },
    "defense": {
        "walk_frames": 0.0764, "squat_frames": 0.1428,
        "stand_frames": 0.0426, "min_speed": 0.2949,
        "min_acceleration": 0.4433
    },
    "decision": {
        "jump_frames": 0.1062, "throw_frames": 0.2605,
        "score_rate": 0.6333
    },
    "agility": {
        "max_speed": 0.2633, "max_acceleration": 0.5579,
        "run_frames": 0.0569, "jump_frames": 0.1219
    },
    "final": {
        "attack": 0.1484, "defense": 0.0876,
        "decision": 0.0490, "agility": 0.0286,
        "total_score": 0.4349, "total_lose_score": 0.2515
    }
}


def compute_static_threat(global_stats):
    """
    计算静态威胁指数
    Args:
        global_stats: 从动态威胁计算中获取的全局统计数据
    Returns:
        dict: {'person_id': {'attack': x, 'defense': y, 'decision': z, 'agility': w, 'threat': v}}
    """
    results = {}

    for person_id, stats in global_stats.items():
        # **归一化数据**
        norm_max_speed = normalize_value(stats["max_speed"], K_VALUES["max_speed"], positive=True)
        norm_max_acceleration = normalize_value(stats["max_acceleration"], K_VALUES["max_acceleration"], positive=True)
        norm_min_speed = normalize_value(stats["min_speed"], K_VALUES["min_speed"], positive=True)
        norm_min_acceleration = normalize_value(stats["min_acceleration"], K_VALUES["min_acceleration"], positive=True)
        norm_score_rate = normalize_value(stats["total_score"], K_VALUES["score_rate"], positive=True)
        norm_lose_rate = normalize_value(stats["total_lose_score"], K_VALUES["lose_rate"], positive=False)

        # **动作统计归一化**
        norm_stand_frames = normalize_value(stats["action_count"]["Stand"], K_VALUES["stand_frames"], positive=True)
        norm_run_frames = normalize_value(stats["action_count"]["Run"], K_VALUES["run_frames"], positive=True)
        norm_jump_frames = normalize_value(stats["action_count"]["Jump"], K_VALUES["jump_frames"], positive=True)
        norm_throw_frames = normalize_value(stats["action_count"]["Throw"], K_VALUES["throw_frames"], positive=True)
        norm_walk_frames = normalize_value(stats["action_count"]["Walk"], K_VALUES["walk_frames"], positive=True)
        norm_squat_frames = normalize_value(stats["action_count"]["Squat"], K_VALUES["squat_frames"], positive=True)

        # **计算四个二级指标**
        attack = (
            WEIGHTS["attack"]["max_speed"] * norm_max_speed +
            WEIGHTS["attack"]["max_acceleration"] * norm_max_acceleration +
            WEIGHTS["attack"]["stand_frames"] * norm_stand_frames +
            WEIGHTS["attack"]["run_frames"] * norm_run_frames +
            WEIGHTS["attack"]["jump_frames"] * norm_jump_frames +
            WEIGHTS["attack"]["throw_frames"] * norm_throw_frames
        )

        defense = (
            WEIGHTS["defense"]["walk_frames"] * norm_walk_frames +
            WEIGHTS["defense"]["squat_frames"] * norm_squat_frames +
            WEIGHTS["defense"]["stand_frames"] * norm_stand_frames +
            WEIGHTS["defense"]["min_speed"] * norm_min_speed +
            WEIGHTS["defense"]["min_acceleration"] * norm_min_acceleration
        )

        decision = (
            WEIGHTS["decision"]["jump_frames"] * norm_jump_frames +
            WEIGHTS["decision"]["throw_frames"] * norm_throw_frames +
            WEIGHTS["decision"]["score_rate"] * norm_score_rate
        )

        agility = (
            WEIGHTS["agility"]["max_speed"] * norm_max_speed +
            WEIGHTS["agility"]["max_acceleration"] * norm_max_acceleration +
            WEIGHTS["agility"]["run_frames"] * norm_run_frames +
            WEIGHTS["agility"]["jump_frames"] * norm_jump_frames
        )

        # **计算最终威胁指数**
        threat_index = (
            WEIGHTS["final"]["attack"] * attack +
            WEIGHTS["final"]["defense"] * defense +
            WEIGHTS["final"]["decision"] * decision +
            WEIGHTS["final"]["agility"] * agility +
            WEIGHTS["final"]["total_score"] * stats["total_score"] +
            WEIGHTS["final"]["total_lose_score"] * stats["total_lose_score"]
        )

        results[person_id] = {
            "attack": round(attack, 4),
            "defense": round(defense, 4),
            "decision": round(decision, 4),
            "agility": round(agility, 4),
            "threat": round(threat_index, 4)
        }

    return results


# **示例调用**
if __name__ == "__main__":
    # **获取动态计算维护的全局统计数据**
    threat_model = ThreatAssessment()
    global_stats = threat_model.get_global_stats()  # 读取全局统计信息
    static_result = compute_static_threat(global_stats)

    # **打印静态威胁指数**
    print("📌 静态威胁指数计算结果:")
    for person_id, values in static_result.items():
        print(f"{person_id}: {values}")
