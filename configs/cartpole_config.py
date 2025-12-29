"""
倒立摆任务配置

任务：控制小车左右移动，保持摆杆直立
"""

CARTPOLE_CONFIG = {
    # ========================================================================
    # 环境配置
    # ========================================================================
    'env_config': {
        'dt': 1.0/60.0,              # 物理时间步长（秒）
        'gravity': 9.81,             # 重力加速度（m/s²）
        'max_steps': 1000,           # 每回合最大步数（增加到1000）
        'force_mag': 10.0,           # 施加力的大小（N）
        'angle_threshold': 12,       # 角度阈值（度）
        'position_threshold': 2.4,   # 位置阈值（米）- 加长轨道
        'disable_termination': True, # 禁用终止条件，让智能体充分探索
        
        # 扰动配置 - 训练时关闭，演示时可开启
        'enable_disturbance': False,  # 训练时关闭扰动
        'disturbance_force_range': 2.0,  # 扰动力范围（N）
        'disturbance_probability': 0.02,  # 每步施加扰动的概率（2%）
        
        # 扰动类型配置
        'disturbance_type': 'cart_only',  # 扰动类型: 'cart_only', 'pole_only', 'both'
        'pole_disturbance_ratio': 0.5,    # 摆杆扰动相对于小车扰动的比例
    },
    
    # ========================================================================
    # PPO算法配置
    # ========================================================================
    'ppo_config': {
        'learning_rate': 3e-4,       # 学习率
        'n_steps': 2048,             # 每次更新前收集的步数
        'batch_size': 64,            # 批量大小
        'n_epochs': 10,              # 每次数据的训练轮数
        'gamma': 0.99,               # 折扣因子
        'gae_lambda': 0.95,          # GAE lambda
        'clip_range': 0.2,           # PPO裁剪范围
        'ent_coef': 0.01,            # 熵系数（探索）
        'vf_coef': 0.5,              # 价值函数系数
        'max_grad_norm': 0.5,        # 梯度裁剪
    },
    
    # ========================================================================
    # 训练流程配置
    # ========================================================================
    'training_config': {
        'n_envs': 32,                 # 并行环境数量
        'n_iterations': 100,         # 总迭代次数（增加以适应更长episode）
        'eval_freq': 10,             # 评估频率（每N次迭代）
        'n_eval_episodes': 5,        # 每次评估的回合数（减少以节省时间）
        'checkpoint_freq': 20,       # 每20次迭代保存一次checkpoint
    },
    
    # ========================================================================
    # 网络架构配置
    # ========================================================================
    'network_config': {
        'net_arch': [64, 64],        # 隐藏层大小（倒立摆较简单，小网络即可）
        'activation_fn': 'tanh',     # 激活函数
    },
    
    # ========================================================================
    # 奖励函数配置
    # ========================================================================
    'reward_config': {
        # 奖励权重在 envs/cartpole/cartpole_env.py 的 _compute_reward 中定义
        'angle_weight': 1.0,         # 角度奖励权重
        'position_weight': 1.0,      # 位置惩罚权重
        'velocity_weight': 0.0,     # 速度惩罚权重
        'alive_weight': 1.0,         # 存活奖励权重
    }
}


# ============================================================================
# 配置说明
# ============================================================================
"""
倒立摆任务特点（增强版）：

1. 训练时间：
   - 通常50-100次迭代收敛
   - 每次迭代需要更长时间（1000步episode）
   - 总训练时间：15-30分钟

2. 鲁棒性测试：
   - 多种扰动类型：cart_only, pole_only, both
   - 可在config文件或命令行中配置扰动参数

3. 成功标准：
   - 保持摆杆直立（角度 < 12°）
   - 小车在轨道范围内（位置 < 4m）
   - 持续1000步不失控
   - 抵抗各种类型的随机扰动

配置方式：

1. 在config文件中修改默认值：
   - 'enable_disturbance': True/False
   - 'disturbance_force_range': 力的大小（N）
   - 'disturbance_probability': 扰动概率（0-1）
   - 'disturbance_type': 扰动类型
   - 'pole_disturbance_ratio': 摆杆扰动比例（仅当type='both'时有效）

2. 命令行参数覆盖config设置：
   --disturbance                    # 启用扰动
   --disturbance-force 3.0          # 扰动力大小
   --disturbance-prob 0.05          # 扰动概率
   --disturbance-type both          # 扰动类型

使用示例：

# 基本扰动测试
python play.py --task cartpole --model model.zip --disturbance

# 自定义扰动参数
python play.py --task cartpole --model model.zip --disturbance --disturbance-type both --disturbance-force 3.0 --disturbance-prob 0.05

# 仅扰动摆杆
python play.py --task cartpole --model model.zip --disturbance --disturbance-type pole_only

# 高频扰动测试
python play.py --task cartpole --model model.zip --disturbance --disturbance-prob 0.1

扰动类型说明：
- cart_only: 🚗 仅扰动小车，摆杆被动响应
- pole_only: 🎯 仅扰动摆杆，小车被动响应  
- both: 🌪️ 同时扰动小车和摆杆，最具挑战性

训练时间估算（8环境，1000步episode）：
- 100次迭代：~15分钟
- 200次迭代：~30分钟（推荐）
- 300次迭代：~45分钟（高鲁棒性）

预期效果：
- 前20次迭代：学会基本平衡
- 20-50次迭代：稳定控制
- 50-100次迭代：精细调优
- 100+次迭代：高质量策略

扰动测试的好处：
- 评估模型鲁棒性
- 发现控制策略的弱点
- 验证在真实环境中的表现
- 对比不同扰动类型的影响
"""
