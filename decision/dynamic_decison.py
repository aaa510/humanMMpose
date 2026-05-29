import numpy as np
import math
from flask import Flask, request, jsonify

# **打击权重（不同难度设定）**
ATTACK_WEIGHTS = {
    "easy": {"TI": 0.7235, "AG": 0.0833, "BoE": 0.1932},
    "medium": {"TI": 0.2972, "AG": 0.1638, "BoE": 0.5390},
    "hard": {"TI": 0.1279, "AG": 0.5120, "BoE": 0.3601}
}

K_AG = 0.4000  # 攻击收益计算的 k 值

app = Flask(__name__)  # **创建 Flask 实例**


class DynamicAttackStrategy:
    def __init__(self):
        """
        初始化动态打击策略，默认难度为 'medium'
        """
        self.difficulty = "medium"  # **默认打击难度**
        self.attack_counts = {}  # **记录每个目标被打击的次数**

    def set_difficulty(self, difficulty):
        """
        允许前端设置打击难度
        Args:
            difficulty (str): 选择 `easy` / `medium` / `hard`
        """
        if difficulty in ATTACK_WEIGHTS:
            self.difficulty = difficulty
            print(f"✅ 设定打击难度为: {difficulty.upper()}")
            return {"status": "success", "message": f"难度设为 {difficulty.upper()}"}
        else:
            print("⚠️ 无效的难度选项，请选择 'easy' / 'medium' / 'hard'")
            return {"status": "error", "message": "无效的难度选项"}

    def compute_boE(self):
        """计算打击平衡性 (BoE) = 被打击次数的标准差"""
        values = np.array(list(self.attack_counts.values()))
        if len(values) < 2:
            return {pid: 0 for pid in self.attack_counts.keys()}  # 仅 1 目标时，BoE 为 0

        std_dev = np.std(values)  # 计算标准差
        return {pid: std_dev for pid in self.attack_counts.keys()}

    def compute_ag(self, defense_score):
        """计算攻击收益 (AG)"""
        return math.exp(-K_AG * defense_score)

    def select_attack_target(self, threat_res):
        """根据 `threat_res` 计算最优打击目标"""

        # **获取当前打击次数 (BoE 计算)**
        for person_id in threat_res.keys():
            if person_id not in self.attack_counts:
                self.attack_counts[person_id] = 0

        boe_scores = self.compute_boE()

        target_scores = {}
        for person_id, values in threat_res.items():
            TI = values["threat"]  # 威胁指数
            AG = self.compute_ag(values["defense"])  # 攻击收益
            BoE = boe_scores.get(person_id, 0)  # 打击平衡性

            # **根据前端选择的难度，调整权重**
            weights = ATTACK_WEIGHTS[self.difficulty]
            attack_priority = (
                weights["TI"] * TI +
                weights["AG"] * AG +
                weights["BoE"] * BoE
            )

            target_scores[person_id] = attack_priority

        # **选择得分最高的目标**
        target_id = max(target_scores, key=target_scores.get)
        self.attack_counts[target_id] += 1  # 目标被攻击次数 +1

        return target_id, target_scores


# ✅ **创建实例**
attack_strategy = DynamicAttackStrategy()


# ----------------------------------------
# **✅ Flask 服务器处理 Vue 前端的请求**
# ----------------------------------------

@app.route('/set-difficulty', methods=['POST'])
def set_difficulty():
    """
    **接收前端 Vue 发送的难度设置信号**
    - POST 传递 JSON：`{"difficulty": "easy"}`
    - 返回 JSON 结果
    """
    data = request.json  # **解析 JSON**
    if not data or "difficulty" not in data:
        return jsonify({"status": "error", "message": "缺少 'difficulty' 字段"}), 400

    difficulty = data["difficulty"].lower()
    response = attack_strategy.set_difficulty(difficulty)
    return jsonify(response)
