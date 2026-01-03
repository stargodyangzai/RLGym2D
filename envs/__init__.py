"""
环境模块 - 包含所有RL任务环境
"""
from .arm.arm_env import ArmEnv
from .walker.walker_env import WalkerEnv
from .cartpole.cartpole_env import CartPoleEnv
from .double_pendulum.double_pendulum_env import DoublePendulumEnv

__all__ = ['ArmEnv', 'WalkerEnv', 'CartPoleEnv', 'DoublePendulumEnv']

# 环境注册表
ENV_REGISTRY = {
    'arm': ArmEnv,
    'walker': WalkerEnv,
    'cartpole': CartPoleEnv,
    'double_pendulum': DoublePendulumEnv,
}

def make_env(env_name, **kwargs):
    """
    创建环境的工厂函数
    
    Args:
        env_name: 环境名称 ('arm', 'walker', 'cartpole', 'double_pendulum')
        **kwargs: 传递给环境的参数
        
    Returns:
        环境实例
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_REGISTRY.keys())}")
    
    return ENV_REGISTRY[env_name](**kwargs)