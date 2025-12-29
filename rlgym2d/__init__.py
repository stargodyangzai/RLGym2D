"""
RLGym2D - 2D Reinforcement Learning Simulation Platform
RLGym2D - 2D强化学习仿真平台

An educational-friendly 2D reinforcement learning platform that serves as a bridge to 3D platforms like Isaac Lab.
一个教育友好的2D强化学习平台，为Isaac Lab等3D平台提供入门桥梁。

Design Philosophy | 设计理念:
- Easy to Learn: Get started in minutes, no complex configuration | 易学易用: 几分钟上手，无需复杂配置
- Intuitive Visualization: 2D visualization, physics processes at a glance | 直观可视: 2D可视化，物理过程一目了然  
- Rapid Prototyping: Quickly validate algorithm ideas | 快速原型: 快速验证算法想法
- Extensible: Easily add new tasks and algorithms | 可扩展: 轻松添加新任务和算法
"""

__version__ = "1.0.0"
__author__ = "RLGym2D Contributors"
__email__ = "rlgym2d@example.com"
__description__ = "RLGym2D - 2D Reinforcement Learning Simulation Platform | 2D强化学习仿真平台"

# Export main modules | 导出主要模块
try:
    from .envs import make_env, ENV_REGISTRY
    from .configs import TASK_CONFIGS
except ImportError:
    # If running as script, use relative imports | 如果作为脚本运行，使用相对导入
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from envs import make_env, ENV_REGISTRY
    from configs import TASK_CONFIGS

__all__ = [
    "make_env",
    "ENV_REGISTRY", 
    "TASK_CONFIGS",
    "__version__",
    "__author__",
    "__description__"
]