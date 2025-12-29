"""
Reward Functions Module - Compatibility Layer
奖励函数模块 - 兼容层

⚠️ This is a deprecated compatibility file
⚠️ 这是旧版本的兼容文件

For new projects, please use:
新项目请使用:
- Robotic Arm: from envs.arm.arm_rewards import *
- 机械臂: from envs.arm.arm_rewards import *
- Humanoid Walker: from envs.walker.walker_rewards import *
- 火柴人: from envs.walker.walker_rewards import *
- Base Classes: from core.base_rewards import RewardFunction, RewardManager
- 通用基类: from core.base_rewards import RewardFunction, RewardManager

This file is only for backward compatibility, it actually imports from core/base_rewards.py
这个文件仅用于向后兼容，实际导入的是 core/base_rewards.py
"""
import warnings

warnings.warn(
    "reward_functions.py is deprecated. Please use 'from core.base_rewards import *' or task-specific reward modules.",
    DeprecationWarning,
    stacklevel=2
)

# Import all contents from core.base_rewards
# 导入所有内容from core.base_rewards
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.base_rewards import *
