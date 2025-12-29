"""
Robotic Arm Environment - Compatibility Layer
机械臂环境 - 兼容层

⚠️ This is a deprecated compatibility file
⚠️ 这是旧版本的兼容文件

For new projects, please use: from envs.arm import ArmEnv
新项目请使用: from envs.arm import ArmEnv

This file is only for backward compatibility, it actually imports from envs/arm/arm_env.py
这个文件仅用于向后兼容，实际导入的是 envs/arm/arm_env.py
"""
import warnings

warnings.warn(
    "simple_arm_env.py is deprecated. Please use 'from envs.arm import ArmEnv' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the new environment class
# 导入新的环境类
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.arm.arm_env import ArmEnv

# Provide old class name as alias
# 提供旧的类名作为别名
SimpleArmEnv = ArmEnv

__all__ = ['SimpleArmEnv', 'ArmEnv']
