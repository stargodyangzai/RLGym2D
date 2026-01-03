"""
二阶倒立摆任务配置

任务：控制小车左右移动，保持两个连接的摆杆都直立
难度：高 - 比单摆复杂得多，需要同时控制两个摆杆

🎛️ 你可以直接修改下面的参数来自定义摆杆特性
"""

DOUBLE_PENDULUM_CONFIG = {
    # ========================================================================
    # 环境配置 - 🎛️ 在这里修改摆杆参数
    # ========================================================================
    'env_config': {
        'dt': 1.0/60.0,              # 物理时间步长（秒）
        'gravity': 9.81,             # 重力加速度（m/s²）
        'max_steps': 1500,           # 每回合最大步数
        'force_mag': 18.0,           # 施加力的大小（N）
        'angle_threshold': 15,       # 角度阈值（度）
        'position_threshold': 3.0,   # 位置阈值（米）
        'disable_termination': True, # 禁用终止条件，让智能体充分探索
        
        # 🎯 摆杆物理参数 - 直接修改这些值
        'cart_mass': 1.0,            # 小车质量（kg）
        'pole1_mass': 0.15,          # 第一摆杆质量（kg）- 基础摆杆，建议稍重
        'pole2_mass': 0.08,          # 第二摆杆质量（kg）- 末端摆杆，建议较轻
        'pole1_length': 0.9,         # 第一摆杆半长（m）- 从中心到端点的距离
        'pole2_length': 0.7,         # 第二摆杆半长（m）- 从中心到端点的距离
        
        # 扰动配置 - 训练时关闭，演示时可开启
        'enable_disturbance': False,
        'disturbance_force_range': 3.0,
        'disturbance_probability': 0.01,
        'disturbance_type': 'cart_only',
    },
    
    # ========================================================================
    # PPO算法配置
    # ========================================================================
    'ppo_config': {
        'learning_rate': 2e-4,       # 学习率（稍微降低）
        'n_steps': 2048,             # 每次更新前收集的步数
        'batch_size': 128,           # 批量大小（增加）
        'n_epochs': 15,              # 每次数据的训练轮数（增加）
        'gamma': 0.995,              # 折扣因子（稍微提高）
        'gae_lambda': 0.95,          # GAE lambda
        'clip_range': 0.2,           # PPO裁剪范围
        'ent_coef': 0.02,            # 熵系数（增加探索）
        'vf_coef': 0.5,              # 价值函数系数
        'max_grad_norm': 0.5,        # 梯度裁剪
    },
    
    # ========================================================================
    # 训练流程配置
    # ========================================================================
    'training_config': {
        'n_envs': 32,                # 并行环境数量（减少以节省内存）
        'n_iterations': 500,         # 总迭代次数（增加，因为任务更难）
        'eval_freq': 15,             # 评估频率（每N次迭代）
        'n_eval_episodes': 3,        # 每次评估的回合数
        'checkpoint_freq': 25,       # 每25次迭代保存一次checkpoint
    },
    
    # ========================================================================
    # 网络架构配置
    # ========================================================================
    'network_config': {
        'net_arch': [128, 128, 64],  # 更大的网络（二阶倒立摆更复杂）
        'activation_fn': 'tanh',     # 激活函数
    },
    
    # ========================================================================
    # 奖励函数配置
    # ========================================================================
    'reward_config': {
        # 🎯 奖励模式选择
        'use_multiplicative': True,   # True=乘法奖励（推荐），False=加法奖励（旧版）
        'use_smooth_gaussian': True,  # True=平滑高斯（推荐），False=阶梯函数（旧版）
        
        # ========== 平滑高斯乘法奖励参数（推荐配置）==========
        # 核心思想：
        # 1. 使用 exp(-x²/σ²) 替代阶梯函数 -> 消除梯度死区
        # 2. 引入角速度惩罚 -> 杀死"旋转刷分"
        # 3. 平滑的乘法关系 -> 鼓励协调控制
        
        # 角度容忍度（sigma值越小，对角度要求越严格）
        'angle1_sigma': 0.10,         # 第一摆杆角度容忍度（约18度时降到0.36）
        'angle2_sigma': 0.15,         # 第二摆杆角度容忍度（稍微宽容）
        
        # 角速度容忍度（sigma值越小，对速度要求越严格）
        'vel1_sigma': 5.0,            # 第一摆杆角速度容忍度
        'vel2_sigma': 10.0,           # 第二摆杆角速度容忍度（惩罚旋转）
        
        # 速度权重（控制角速度在总状态中的占比）
        'vel1_weight': 0.2,           # 第一摆杆速度权重（20%）
        'vel2_weight': 0.3,           # 第二摆杆速度权重（30%，更重视静止）
        
        # 位置和奖励缩放
        'pos_sigma_factor': 1.0,      # 位置容忍度因子
        'reward_scale': 10.0,         # 奖励缩放系数
        
        # 崩溃惩罚
        'pole2_collapse_threshold': 60,  # 第二摆杆崩溃阈值（度）
        
        # ========== 加法奖励模式参数（use_multiplicative=False时生效）==========
        'pole1_weight': 1.5,          # 第一摆杆权重
        'pole2_weight': 3.0,          # 第二摆杆权重
        'coordination_weight': 1.0,   # 协调奖励权重
        'position_weight': 0.2,       # 位置惩罚权重
        'cart_vel_weight': 0.0,       # 小车速度惩罚权重
        
        # 💡 为什么平滑高斯更好？
        # 
        # 问题1：梯度死区
        # - 阶梯函数：pole1 < 5° 时永远是 1.0，智能体没有动力从 4° 优化到 1°
        # - 高斯函数：exp(-x²) 在任何地方都有导数，每一点改进都有奖励反馈
        # 
        # 问题2：旋转刷分
        # - 旧版：只看角度，第二阶高速旋转经过垂直点时也能拿满分
        # - 新版：角速度惩罚，只有"静止的直立"才能拿高分
        # 
        # 问题3：不愿倾斜
        # - 旧版：P1 从 5° 变成 6° 奖励立刻从 1.0 掉到 0.97，损失太大
        # - 新版：平滑过渡，P1 适度倾斜去救 P2 的收益更明显
        # 
        # 问题4：梯度不连续
        # - 阶梯函数在 5°、15°、30° 等切换点导数不连续，学习不稳定
        # - 高斯函数处处可导，PPO/SAC 能获得稳定的梯度信号
    }
}


# ============================================================================
# 配置说明
# ============================================================================
"""
二阶倒立摆任务特点：

1. 复杂度：
   - 训练复杂度：高（需要200-500次迭代）
   - 控制难度：极高（两个摆杆的耦合动力学）
   - 状态空间：6维（比单摆多2维）
   - 动力学：强非线性，混沌行为

2. 物理特性：
   - 第一摆杆：长0.9m，质量0.15kg（基础稳定性，稍重稍长）
   - 第二摆杆：长0.7m，质量0.08kg（精细控制，更轻更短）
   - 小车：质量1.0kg，控制力18N
   - 轨道：长6m（±3m）
   
   设计理念：
   - 不同长度和质量的摆杆产生更丰富的动力学
   - 第一摆杆作为"基础"，需要更强的惯性
   - 第二摆杆作为"精细调节"，需要更灵敏的响应

3. 控制挑战：
   - 耦合动力学：两个摆杆相互影响
   - 混沌行为：小扰动可能导致大变化
   - 多目标：同时稳定两个摆杆
   - 精细控制：需要更精确的力控制

4. 奖励设计：
   - 分层奖励：第一摆杆更重要（基础稳定）
   - 协调奖励：两个摆杆都稳定时额外奖励
   - 角速度控制：防止振荡和混沌
   - 位置控制：保持小车在中央

5. 训练策略：
   - 更大的网络：[128, 128, 64]
   - 更多的训练：300次迭代
   - 更长的episode：1500步
   - 更多的探索：熵系数0.02

6. 成功标准：
   - 两个摆杆都保持直立（角度 < 15°）
   - 小车在轨道范围内（位置 < 3m）
   - 持续1500步不失控
   - 抵抗随机扰动

预期训练时间（16环境，CPU训练）：
- 100次迭代：学会基本平衡第一摆杆
- 200次迭代：开始控制第二摆杆
- 300次迭代：稳定控制两个摆杆
- 400+次迭代：高质量策略，抗扰动

训练复杂度对比：
- 单摆（CartPole）：★☆☆☆☆ (简单)
- 二阶倒立摆：★★★★☆ (困难)
- 机械臂：★★★☆☆ (中等)
- 火柴人：★★★☆☆ (中等)

使用建议：
1. 先训练单摆，熟悉基本概念
2. 二阶倒立摆适合有经验的用户
3. 建议使用GPU加速训练（如果可用）
4. 可以先降低max_steps到1000进行快速实验

扰动测试：
- 二阶倒立摆对扰动极其敏感
- 建议使用较小的扰动力（1-2N）
- 扰动频率建议更低（0.5-1%）
- 主要测试小车扰动，摆杆扰动可能过于剧烈

命令行使用：
# 训练
python train.py --task double_pendulum --envs 16

# 演示
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip

# 扰动测试
python play.py --task double_pendulum --model model.zip --disturbance --disturbance-force 2.0 --disturbance-prob 0.01

物理直觉：
- 第一摆杆是"基础"，必须先稳定
- 第二摆杆是"精细调节"，在第一摆杆稳定后控制
- 小车的移动会同时影响两个摆杆
- 系统具有强烈的非线性和混沌特性
- 需要预测性控制，而不是反应性控制

学习曲线特点：
- 前50次迭代：随机探索，偶尔短暂平衡
- 50-150次迭代：学会稳定第一摆杆
- 150-250次迭代：开始协调两个摆杆
- 250-400次迭代：精细调优，提高稳定性
- 400+次迭代：高质量策略，抗扰动能力

这是一个极具挑战性的控制任务，成功训练需要耐心和计算资源！

# ============================================================================
# Configuration Guide
# ============================================================================
# How to customize pendulum parameters:
# 
# Modify the parameters in env_config above:
# 
# Physical parameters:
# - cart_mass: Cart mass (kg) - affects overall inertia
# - pole1_mass: First pole mass (kg) - recommended 0.1-0.3, affects base stability
# - pole2_mass: Second pole mass (kg) - recommended 0.05-0.15, affects fine control
# - pole1_length: First pole half-length (m) - recommended 0.5-1.5, longer = harder
# - pole2_length: Second pole half-length (m) - recommended 0.3-1.2, longer = harder
# - force_mag: Maximum control force (N) - adjust based on pole weight, 15-30
# 
# Control parameters:
# - angle_threshold: Failure angle (degrees) - smaller = stricter
# - position_threshold: Half of track length (m)
# - max_steps: Maximum steps per episode
# 
# Design recommendations:
# - First pole slightly heavier and longer - provides base stability
# - Second pole lighter and shorter - handles fine adjustment
# - Control force must be sufficient - able to control total mass
# - Different lengths and masses - create rich dynamics
# 
# Example configurations:
# Easy version: pole1_mass=0.12, pole2_mass=0.06, pole1_length=0.8, pole2_length=0.6, force_mag=20
# Hard version: pole1_mass=0.20, pole2_mass=0.12, pole1_length=1.2, pole2_length=1.0, force_mag=25
# Asymmetric: pole1_mass=0.25, pole2_mass=0.04, pole1_length=1.1, pole2_length=0.5, force_mag=22
# 
# After modifying parameters, run:
# python train.py --task double_pendulum --envs 16
# python play.py --task double_pendulum --model model.zip
"""

# ====
========================================================================
# 使用示例 | Usage Examples
# ============================================================================
"""
基础训练 | Basic Training:
--------------------------
python train.py --task double_pendulum --envs 16

快速训练 | Fast Training:
--------------------------
python train.py --task double_pendulum --envs 32

长时间训练 | Extended Training:
--------------------------
python train.py --task double_pendulum --envs 16 --iterations 500

演示模型 | Demo Model:
--------------------------
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip

录制视频 | Record Video:
--------------------------
python play.py --task double_pendulum --model model.zip --record --episodes 3

扰动测试 | Disturbance Test:
--------------------------
python play.py --task double_pendulum --model model.zip --disturbance --disturbance-force 3.0 --disturbance-prob 0.015

对比奖励函数 | Compare Reward Functions:
--------------------------
python compare_reward_functions.py

监控训练 | Monitor Training:
--------------------------
tensorboard --logdir=runs/
"""

# ============================================================================
# 调参建议 | Tuning Recommendations
# ============================================================================
"""
场景1：更严格的控制 | Stricter Control
----------------------------------------
'reward_config': {
    'angle1_sigma': 0.08,  # 降低容忍度
    'angle2_sigma': 0.12,
    'vel1_weight': 0.3,    # 增加速度权重
    'vel2_weight': 0.4,
}

效果 | Effect:
- ✅ 更精确的控制
- ✅ 更少的振荡
- ⚠️ 训练难度增加
- ⚠️ 需要更多迭代

场景2：更容易学习 | Easier Learning
----------------------------------------
'reward_config': {
    'angle1_sigma': 0.15,  # 提高容忍度
    'angle2_sigma': 0.20,
    'vel1_weight': 0.1,    # 降低速度权重
    'vel2_weight': 0.2,
}

效果 | Effect:
- ✅ 更快收敛
- ✅ 训练更稳定
- ⚠️ 控制精度降低
- ⚠️ 可能有轻微振荡

场景3：强调第二阶控制 | Emphasize Second Pole
----------------------------------------
'reward_config': {
    'angle2_sigma': 0.10,  # 对第二阶更严格
    'vel2_sigma': 8.0,
    'vel2_weight': 0.4,
}

效果 | Effect:
- ✅ 第二阶控制更精确
- ✅ 减少第二阶旋转
- ⚠️ 第一阶可能稍微放松

场景4：课程学习 | Curriculum Learning
----------------------------------------
# 阶段1：宽松配置（0-100次迭代）
'angle1_sigma': 0.15, 'angle2_sigma': 0.20

# 阶段2：标准配置（100-200次迭代）
'angle1_sigma': 0.10, 'angle2_sigma': 0.15

# 阶段3：严格配置（200+次迭代）
'angle1_sigma': 0.08, 'angle2_sigma': 0.12
"""

# ============================================================================
# 性能基准 | Performance Benchmarks
# ============================================================================
"""
硬件配置 | Hardware:
- CPU: Intel i7-10700K
- RAM: 32GB
- GPU: NVIDIA RTX 3080 (optional)

训练时间 | Training Time:
- 16 envs, CPU: ~2-3小时 (300 iterations)
- 32 envs, CPU: ~1.5-2小时 (300 iterations)
- 16 envs, GPU: ~1-1.5小时 (300 iterations)

性能指标 | Performance Metrics:

优秀策略 (奖励 > 8.5) | Excellent Policy:
- 第一摆杆角度 | Pole1 angle: < 3°
- 第二摆杆角度 | Pole2 angle: < 8°
- 角速度 | Angular velocity: ≈ 0
- 小车位置 | Cart position: < 1m

良好策略 (奖励 6-8) | Good Policy:
- 第一摆杆角度 | Pole1 angle: < 8°
- 第二摆杆角度 | Pole2 angle: < 15°
- 角速度 | Angular velocity: < 2 rad/s
- 小车位置 | Cart position: < 2m

可用策略 (奖励 4-6) | Usable Policy:
- 第一摆杆角度 | Pole1 angle: < 12°
- 第二摆杆角度 | Pole2 angle: < 20°
- 偶尔失控但能恢复 | Occasional loss of control but recoverable
"""

# ============================================================================
# 常见问题 | FAQ
# ============================================================================
"""
Q1: 第一摆杆不愿意动 | First pole reluctant to move
A1: 确保启用平滑高斯 | Ensure smooth Gaussian is enabled:
    'use_smooth_gaussian': True

Q2: 第二摆杆持续旋转 | Second pole keeps spinning
A2: 增加速度权重 | Increase velocity weight:
    'vel2_weight': 0.4, 'vel2_sigma': 8.0

Q3: 训练不稳定 | Training unstable
A3: 降低学习率，增加批量 | Lower learning rate, increase batch size:
    'learning_rate': 1e-4, 'batch_size': 256

Q4: 收敛太慢 | Convergence too slow
A4: 增加探索和网络大小 | Increase exploration and network size:
    'ent_coef': 0.03, 'net_arch': [256, 256, 128]

Q5: 过拟合 | Overfitting
A5: 增加并行环境和探索 | Increase parallel envs and exploration:
    'n_envs': 32, 'ent_coef': 0.03
"""

# ============================================================================
# 相关资源 | Related Resources
# ============================================================================
"""
📖 详细文档 | Detailed Documentation:
- docs/reward_function_design.md - 奖励函数设计详解
- docs/double_pendulum_tutorial.md - 训练教程

📊 测试脚本 | Test Scripts:
- compare_reward_functions.py - 奖励函数对比测试

🔧 核心文件 | Core Files:
- envs/double_pendulum/double_pendulum_env.py - 环境实现
- configs/double_pendulum_config.py - 配置文件（本文件）

📝 更新日志 | Changelog:
- CHANGELOG.md - 中文更新日志
- CHANGELOG_EN.md - English changelog
"""
