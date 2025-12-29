"""
奖励函数模块 - 兼容层

⚠️ 这是旧版本的兼容文件
新项目请使用:
- 机械臂: from envs.arm.arm_rewards import *
- 火柴人: from envs.walker.walker_rewards import *
- 通用基类: from core.base_rewards import RewardFunction, RewardManager

这个文件仅用于向后兼容，实际导入的是 core/base_rewards.py
"""
import warnings

warnings.warn(
    "reward_functions.py is deprecated. Please use 'from core.base_rewards import *' or task-specific reward modules.",
    DeprecationWarning,
    stacklevel=2
)

# 导入所有内容from core.base_rewards
from core.base_rewards import *
