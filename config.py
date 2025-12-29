"""
配置文件 - 兼容层

⚠️ 这是旧版本的兼容文件
新项目请使用: from configs import TASK_CONFIGS

这个文件仅用于向后兼容，实际导入的是 configs/arm_config.py
"""
import warnings

warnings.warn(
    "config.py is deprecated. Please use 'from configs import TASK_CONFIGS' or 'from configs.arm_config import ARM_CONFIG' instead.",
    DeprecationWarning,
    stacklevel=2
)

# 导入新的配置
from configs.arm_config import ARM_CONFIG

# 提供旧的变量名
ENV_CONFIG = ARM_CONFIG['env_config']
REWARD_CONFIG = ARM_CONFIG['reward_config']
PPO_CONFIG = ARM_CONFIG['ppo_config']
TRAINING_CONFIG = ARM_CONFIG['training_config']
NETWORK_CONFIG = ARM_CONFIG['network_config']
DEVICE_CONFIG = {'device': 'auto'}
WANDB_CONFIG = {
    'enabled': True,
    'project': '2d-arm-rl',
    'entity': None,
    'name': None,
    'tags': ['ppo', '2d-arm', 'robotics'],
    'notes': '',
    'sync_tensorboard': True,
}

def get_config(preset=None):
    """
    获取配置（兼容旧版本）
    
    ⚠️ 已弃用，请使用 configs.arm_config.ARM_CONFIG
    """
    import torch.nn as nn
    
    config = {
        'env': ENV_CONFIG.copy(),
        'reward': REWARD_CONFIG.copy(),
        'ppo': PPO_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'network': NETWORK_CONFIG.copy(),
        'device': DEVICE_CONFIG.copy(),
        'wandb': WANDB_CONFIG.copy(),
    }
    
    # 转换激活函数
    activation_fn_str = config['network']['policy_kwargs'].get('activation_fn', 'tanh')
    activation_fn_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'leaky_relu': nn.LeakyReLU,
    }
    config['network']['policy_kwargs']['activation_fn'] = activation_fn_map.get(
        activation_fn_str, nn.Tanh
    )
    
    return config

def print_config(config=None):
    """打印配置（兼容旧版本）"""
    if config is None:
        config = get_config()
    
    print("\n" + "="*70)
    print("当前配置（使用旧版本兼容层）")
    print("="*70)
    print("⚠️ 建议使用新版本: from configs.arm_config import ARM_CONFIG")
    print("="*70)
    
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    print("="*70 + "\n")

__all__ = [
    'ENV_CONFIG', 'REWARD_CONFIG', 'PPO_CONFIG', 'TRAINING_CONFIG',
    'NETWORK_CONFIG', 'DEVICE_CONFIG', 'WANDB_CONFIG',
    'get_config', 'print_config'
]
