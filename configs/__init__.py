"""
配置模块 - 管理所有任务的配置
"""
from .arm_config import ARM_CONFIG
from .walker_config import WALKER_CONFIG
from .cartpole_config import CARTPOLE_CONFIG
from .double_pendulum_config import DOUBLE_PENDULUM_CONFIG

# 任务配置注册表
TASK_CONFIGS = {
    'arm': ARM_CONFIG,
    'walker': WALKER_CONFIG,
    'cartpole': CARTPOLE_CONFIG,
    'double_pendulum': DOUBLE_PENDULUM_CONFIG,
}

__all__ = ['TASK_CONFIGS', 'ARM_CONFIG', 'WALKER_CONFIG', 'CARTPOLE_CONFIG', 'DOUBLE_PENDULUM_CONFIG']
