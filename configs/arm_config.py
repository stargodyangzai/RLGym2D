"""
机械臂任务配置

任务：2D双关节机械臂到达目标点

这个配置文件从原 config.py 迁移而来，保持所有原有参数不变。
"""

ARM_CONFIG = {
    # ========================================================================
    # 环境配置 (Environment Parameters)
    # ========================================================================
    'env_config': {
        # 物理仿真
        'dt': 1.0 / 60.0,  # 物理仿真时间步长 (秒)
        'gravity': 9.81,  # 重力加速度 (m/s²), 0=无重力, 9.81=标准重力
        'max_steps': 500,  # 每个回合的最大步数
        
        # 机械臂物理属性（公制单位）
        'link_lengths': [1.0, 0.8],  # 各连杆长度 (米)
        'link_mass': 0.3,  # 每段连杆质量 (kg)
        'max_torque': 300.0,  # 关节最大扭矩 (N·m)
        
        # 运动控制
        'angular_velocity_limit': 20.0,  # 最大角速度限制 (rad/s)
        'velocity_damping': 0.9,  # 线速度阻尼系数 (0-1)
        'angular_velocity_damping': 0.9,  # 角速度阻尼系数 (0-1)
        
        # 目标生成（公制单位）
        'target_distance_min': 0.5,  # 目标最小距离 (米)
        'target_distance_max': 1.5,  # 目标最大距离 (米)
        'target_angle_min': 0,  # 目标角度最小值 (弧度)
        'target_angle_max': 2*3.1415926535,  # 目标角度最大值 (弧度)
        
        # 渲染参数
        'pixels_per_meter': 100,  # 渲染比例：100像素 = 1米
    },
    
    # ========================================================================
    # 奖励函数配置 (Reward Configuration)
    # ========================================================================
    'reward_config': {
        # 奖励模式选择
        'mode': 'smooth',  # 'default', 'smooth', 'efficient', 'advanced', 'custom'
        
        # 距离奖励类型
        'use_squared_distance': False,  # 改回线性，学习信号更清晰
        
        # 基础奖励参数（公制单位）
        'distance_weight': 1.0,  # 线性距离奖励（单位：米）
        'success_weight': 1500.0,  # 成功奖励（大幅增加，补偿所有惩罚）
        'failure_penalty_weight': 0.0,  # 失败惩罚（暂时关闭）
        'success_threshold': 0.15,  # 成功阈值（米，15厘米）
        
        # 平滑性奖励参数（mode='smooth' 或 'advanced' 时启用）
        'velocity_penalty_weight': 0.1,  # 速度惩罚（大幅降低，允许快速调整方向）
        'action_penalty_weight': 0.2,  # 动作惩罚（大幅降低，鼓励改变方向）
        'acceleration_penalty_weight': 0.0,  # 加速度惩罚（暂时关闭）
        
        # 效率奖励参数（mode='efficient' 或 'advanced' 时启用）
        'progress_weight': 0.5,  # 进度奖励（奖励接近，惩罚远离，引导选择最短路径）
        'time_efficiency_weight': 0.0,  # 时间效率权重（暂时关闭）
        
        # 高级奖励参数（mode='advanced' 时可选）
        'orientation_weight': 0.0,  # 方向奖励权重（末端朝向目标）- 暂时关闭
        
        # 到达质量奖励参数（新增）
        'arrival_velocity_penalty_weight': 200.0,  # 到达速度惩罚（惩罚高速到达）
        'path_efficiency_penalty_weight': 1000.0,  # 路径效率惩罚（惩罚绕路，只在成功时计算）
        'path_length_penalty_weight': 2.0,  # 路径长度惩罚（每步惩罚移动距离，持续反馈）
        
        # 自定义奖励组件（mode='custom' 时使用）
        'custom_components': {},
    },
    
    # ========================================================================
    # PPO算法配置 (PPO Hyperparameters)
    # ========================================================================
    'ppo_config': {
        'policy': 'MlpPolicy',  # 策略网络类型
        'learning_rate': 1e-5,  # 学习率 (典型范围: 1e-5 到 1e-3)
        'n_steps': 1024,  # 每次策略更新前收集的环境步数
        'batch_size': 64,  # 小批量大小
        'n_epochs': 10,  # 每次收集数据后的训练轮数
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE lambda 参数
        'clip_range': 0.2,  # PPO 裁剪范围
        'clip_range_vf': None,  # 价值函数裁剪范围 (None=不裁剪)
        'ent_coef': 0.005,  # 熵系数 (增加探索，鼓励尝试不同方向)
        'vf_coef': 0.5,  # 价值函数损失系数
        'max_grad_norm': 0.5,  # 梯度裁剪阈值
        'verbose': 1,  # 日志详细程度 (0=无, 1=训练信息, 2=调试)
    },
    
    # ========================================================================
    # 训练流程配置 (Training Configuration)
    # ========================================================================
    'training_config': {
        'n_envs': 32,  # 并行环境数量
        'n_iterations': 100,  # PPO迭代次数
        'eval_freq': 10,  # 评估频率（每N次迭代）
        'n_eval_episodes': 100,  # 每次评估的回合数
        'checkpoint_freq': 10,  # 每10次迭代保存一次checkpoint
    },
    
    # ========================================================================
    # 网络架构配置 (Network Architecture)
    # ========================================================================
    'network_config': {
        'net_arch': [64, 32],  # 神经网络隐藏层大小 [层1, 层2, ...]
        'activation_fn': 'tanh',  # 激活函数: 'tanh', 'relu', 'elu', 'leaky_relu'
    },
    
    # ========================================================================
    # 设备配置 (Device Configuration)
    # ========================================================================
    'device_config': {
        'device': 'auto',  # 训练设备: 'auto', 'cuda', 'cpu'
    },
    
    # ========================================================================
    # Weights & Biases 配置 (W&B Configuration)
    # ========================================================================
    'wandb_config': {
        'enabled': True,  # 是否启用 W&B 日志记录
        'project': '2d-arm-rl',  # W&B 项目名称
        'entity': None,  # W&B 用户名或团队名 (None=使用默认)
        'name': None,  # 运行名称 (None=自动生成)
        'tags': ['ppo', '2d-arm', 'robotics'],  # 标签列表
        'notes': '',  # 运行说明
        'sync_tensorboard': True,  # 是否同步 TensorBoard 日志到 W&B
    },
    
    # ========================================================================
    # 评估配置 (Evaluation Configuration)
    # ========================================================================
    'eval_config': {
        'n_episodes': 100,  # 默认评估回合数
        'render': False,  # 是否显示可视化
        'deterministic': True,  # 是否使用确定性策略
    },
}


# ============================================================================
# 快速调优建议 (Quick Tuning Guide)
# ============================================================================
"""
常见调优场景：

1. 机械臂运动太慢：
   - 增加 env_config['max_torque'] (200 → 300)
   - 减小 env_config['link_mass'] (0.5 → 0.3)

2. 机械臂运动不稳定/震荡：
   - 增加 env_config['velocity_damping'] (0.9 → 0.95)
   - 增加 env_config['angular_velocity_damping'] (0.9 → 0.95)
   - 减小 env_config['angular_velocity_limit'] (20 → 15)

3. 训练速度慢：
   - 增加 training_config['n_envs'] (32 → 64)
   - 减小 ppo_config['n_steps'] (1024 → 512)

4. 训练不稳定/不收敛：
   - 减小 ppo_config['learning_rate'] (1e-5 → 1e-6)
   - 增加 ppo_config['n_epochs'] (10 → 20)
   - 增加 ppo_config['batch_size'] (64 → 128)

5. 模型不够探索：
   - 增加 ppo_config['ent_coef'] (0.005 → 0.01)

6. 需要更大的网络：
   - 修改 network_config['net_arch'] ([64,32] → [128,64])

7. 任务太难/太简单：
   - 调整 env_config['target_distance_min/max']
   - 调整 reward_config['success_threshold']
   - 调整 env_config['max_steps']

8. 调整奖励函数：
   - 调整 reward_config['distance_weight'] 改变距离奖励权重
   - 调整 reward_config['success_weight'] 改变成功奖励大小
   - 调整 reward_config['success_threshold'] 改变成功判定标准
"""
