"""
机械臂环境 - 兼容层

⚠️ 这是旧版本的兼容文件
新项目请使用: from envs.arm import ArmEnv

这个文件仅用于向后兼容，实际导入的是 envs/arm/arm_env.py
"""
import warnings

warnings.warn(
    "simple_arm_env.py is deprecated. Please use 'from envs.arm import ArmEnv' instead.",
    DeprecationWarning,
    stacklevel=2
)

# 导入新的环境类
from envs.arm.arm_env import ArmEnv

# 提供旧的类名作为别名
SimpleArmEnv = ArmEnv

__all__ = ['SimpleArmEnv', 'ArmEnv']
