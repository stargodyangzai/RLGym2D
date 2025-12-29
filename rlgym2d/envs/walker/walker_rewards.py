"""
火柴人行走任务的奖励函数

这个文件定义了火柴人行走任务的奖励组件。
目前奖励函数直接在 walker_env.py 的 _compute_reward 方法中实现。

未来可以在这里定义独立的奖励组件类，类似机械臂的奖励函数。
"""

# 奖励权重配置（可以从 configs/walker_config.py 读取）
DEFAULT_REWARD_WEIGHTS = {
    'forward_weight': 0.1,       # 前进奖励权重
    'velocity_weight': 0.5,      # 速度奖励权重
    'alive_weight': 1.0,         # 存活奖励权重
    'balance_weight': 2.0,       # 平衡惩罚权重
    'height_weight': 0.5,        # 高度奖励权重
    'action_weight': 0.01,       # 动作惩罚权重
}


# TODO: 未来可以实现独立的奖励组件类
# 例如：
# class ForwardReward(RewardFunction):
#     """前进奖励 - 鼓励向右移动"""
#     pass
#
# class BalancePenalty(RewardFunction):
#     """平衡惩罚 - 惩罚倾斜"""
#     pass
