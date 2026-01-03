# 二阶倒立摆训练教程

## 🎯 快速开始

### 1. 基础训练

```bash
# 使用默认配置训练（推荐）
python train.py --task double_pendulum --envs 16

# 使用更多并行环境加速训练
python train.py --task double_pendulum --envs 32

# 更长的训练获得更好的策略
python train.py --task double_pendulum --envs 16 --iterations 500
```

### 2. 演示模型

```bash
# 基础演示
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip

# 录制视频
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip --record --episodes 3

# 扰动测试（测试鲁棒性）
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip --disturbance --disturbance-force 3.0 --disturbance-prob 0.015
```

---

## 📊 训练监控

### 关键指标

在 TensorBoard 中关注以下指标：

```bash
tensorboard --logdir=runs/
```

**奖励指标**：
- `rollout/ep_rew_mean`: 平均回合奖励
  - 目标：> 8.0（优秀）
  - 目标：> 6.0（良好）
  - 目标：> 4.0（可用）

**训练指标**：
- `train/policy_gradient_loss`: 策略梯度损失
- `train/value_loss`: 价值函数损失
- `train/entropy_loss`: 熵损失（探索程度）

**自定义指标**（二阶倒立摆特有）：
- `custom/pole1_angle_deg`: 第一摆杆平均角度
- `custom/pole2_angle_deg`: 第二摆杆平均角度
- `custom/p1_angle_status`: 第一摆杆角度状态
- `custom/p2_angle_status`: 第二摆杆角度状态
- `custom/v1_status`: 第一摆杆速度状态
- `custom/v2_status`: 第二摆杆速度状态

---

## 🔧 配置调优

### 奖励函数参数

编辑 `configs/double_pendulum_config.py`：

#### 场景1：更严格的控制

```python
'reward_config': {
    'use_multiplicative': True,
    'use_smooth_gaussian': True,
    
    # 降低角度容忍度
    'angle1_sigma': 0.08,  # 默认 0.10
    'angle2_sigma': 0.12,  # 默认 0.15
    
    # 增加速度权重
    'vel1_weight': 0.3,    # 默认 0.2
    'vel2_weight': 0.4,    # 默认 0.3
}
```

**效果**：
- ✅ 更精确的控制
- ✅ 更少的振荡
- ⚠️ 训练难度增加
- ⚠️ 需要更多迭代

#### 场景2：更容易学习

```python
'reward_config': {
    'use_multiplicative': True,
    'use_smooth_gaussian': True,
    
    # 提高角度容忍度
    'angle1_sigma': 0.15,  # 默认 0.10
    'angle2_sigma': 0.20,  # 默认 0.15
    
    # 降低速度权重
    'vel1_weight': 0.1,    # 默认 0.2
    'vel2_weight': 0.2,    # 默认 0.3
}
```

**效果**：
- ✅ 更快收敛
- ✅ 训练更稳定
- ⚠️ 控制精度降低
- ⚠️ 可能有轻微振荡

#### 场景3：强调第二阶控制

```python
'reward_config': {
    'use_multiplicative': True,
    'use_smooth_gaussian': True,
    
    # 对第二阶更严格
    'angle2_sigma': 0.10,  # 默认 0.15
    'vel2_sigma': 8.0,     # 默认 10.0
    'vel2_weight': 0.4,    # 默认 0.3
}
```

**效果**：
- ✅ 第二阶控制更精确
- ✅ 减少第二阶旋转
- ⚠️ 第一阶可能稍微放松

### PPO 算法参数

```python
'ppo_config': {
    # 学习率调整
    'learning_rate': 2e-4,    # 默认值，可以尝试 1e-4（更稳定）或 3e-4（更快）
    
    # 批量大小
    'batch_size': 128,        # 默认值，可以尝试 256（更稳定）或 64（更快）
    
    # 训练轮数
    'n_epochs': 15,           # 默认值，可以尝试 20（更充分）或 10（更快）
    
    # 熵系数（探索）
    'ent_coef': 0.02,         # 默认值，可以尝试 0.01（更少探索）或 0.03（更多探索）
}
```

---

## 📈 训练阶段分析

### 阶段1：探索（0-50次迭代）

**特征**：
- 奖励：0-3
- 行为：随机探索，偶尔短暂平衡
- 第一摆杆：大幅摆动
- 第二摆杆：混乱运动

**正常现象**：
- ✅ 奖励波动大
- ✅ 成功率接近 0
- ✅ 角度变化剧烈

**异常现象**：
- ⚠️ 奖励一直为负（检查环境配置）
- ⚠️ 完全不动（检查动作空间）

### 阶段2：学习第一摆杆（50-150次迭代）

**特征**：
- 奖励：3-5
- 行为：开始稳定第一摆杆
- 第一摆杆：角度 < 20°
- 第二摆杆：仍然不稳定

**正常现象**：
- ✅ 第一摆杆角度逐渐减小
- ✅ 奖励稳步上升
- ✅ 第二摆杆还在摆动

**异常现象**：
- ⚠️ 第一摆杆不动（可能陷入局部最优）
- ⚠️ 奖励停滞（尝试增加熵系数）

### 阶段3：协调控制（150-250次迭代）⭐ 关键

**特征**：
- 奖励：5-8
- 行为：开始协调控制两个摆杆
- 第一摆杆：会适度倾斜去救第二摆杆
- 第二摆杆：开始稳定

**正常现象**：
- ✅ 第一摆杆不再"僵硬"
- ✅ 第二摆杆旋转减少
- ✅ 奖励快速增长

**这是平滑高斯奖励的优势体现阶段！**

**异常现象**：
- ⚠️ 第一摆杆死守 0°（检查奖励函数配置）
- ⚠️ 第二摆杆持续旋转（增加 vel2_weight）

### 阶段4：精细调优（250-400次迭代）

**特征**：
- 奖励：8-10
- 行为：两个摆杆都稳定直立
- 第一摆杆：角度 < 5°
- 第二摆杆：角度 < 10°，不旋转

**正常现象**：
- ✅ 奖励接近最大值
- ✅ 控制平滑
- ✅ 抗扰动能力强

**可以停止训练的标志**：
- ✅ 平均奖励 > 8.5
- ✅ 评估成功率 > 90%
- ✅ 奖励曲线平稳

---

## 🐛 常见问题

### Q1: 第一摆杆不愿意动

**症状**：第一摆杆死守 0°，即使第二摆杆倒了也不动

**原因**：可能使用了阶梯式奖励函数

**解决**：
```python
# 确保启用平滑高斯
'use_smooth_gaussian': True,
```

### Q2: 第二摆杆持续旋转

**症状**：第二摆杆高速旋转，经过垂直点时拿分

**原因**：角速度惩罚不够

**解决**：
```python
# 增加速度权重
'vel2_weight': 0.4,  # 从 0.3 增加到 0.4
'vel2_sigma': 8.0,   # 从 10.0 降低到 8.0
```

### Q3: 训练不稳定

**症状**：奖励剧烈波动，时好时坏

**原因**：学习率太高或批量太小

**解决**：
```python
'learning_rate': 1e-4,  # 降低学习率
'batch_size': 256,      # 增加批量大小
```

### Q4: 收敛太慢

**症状**：300次迭代后奖励还是很低

**原因**：探索不足或网络太小

**解决**：
```python
'ent_coef': 0.03,           # 增加探索
'net_arch': [256, 256, 128], # 增大网络
```

### Q5: 过拟合

**症状**：训练奖励高，但评估奖励低

**原因**：过度拟合训练环境

**解决**：
```python
'n_envs': 32,        # 增加并行环境
'ent_coef': 0.03,    # 增加探索
```

---

## 🎓 进阶技巧

### 1. 课程学习

逐步增加难度：

```python
# 阶段1：宽松配置（0-100次迭代）
'angle1_sigma': 0.15,
'angle2_sigma': 0.20,

# 阶段2：标准配置（100-200次迭代）
'angle1_sigma': 0.10,
'angle2_sigma': 0.15,

# 阶段3：严格配置（200+次迭代）
'angle1_sigma': 0.08,
'angle2_sigma': 0.12,
```

### 2. 继续训练

从已有模型继续训练：

```bash
python continue_train.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip --iterations 200
```

### 3. 对比实验

测试不同奖励函数：

```bash
# 实验1：加法奖励
# 修改配置：use_multiplicative = False
python train.py --task double_pendulum --envs 16

# 实验2：阶梯式乘法
# 修改配置：use_multiplicative = True, use_smooth_gaussian = False
python train.py --task double_pendulum --envs 16

# 实验3：平滑高斯乘法（推荐）
# 修改配置：use_multiplicative = True, use_smooth_gaussian = True
python train.py --task double_pendulum --envs 16
```

### 4. 扰动鲁棒性训练

在训练时启用扰动：

```python
'env_config': {
    'enable_disturbance': True,
    'disturbance_force_range': 2.0,
    'disturbance_probability': 0.01,
}
```

---

## 📊 性能基准

### 硬件配置

- CPU: Intel i7-10700K
- RAM: 32GB
- GPU: NVIDIA RTX 3080 (可选)

### 训练时间

| 配置 | 迭代次数 | 时间 | 最终奖励 |
|------|---------|------|---------|
| 16 envs, CPU | 300 | ~2-3小时 | 8.0-8.5 |
| 32 envs, CPU | 300 | ~1.5-2小时 | 8.0-8.5 |
| 16 envs, GPU | 300 | ~1-1.5小时 | 8.0-8.5 |

### 性能指标

**优秀策略**（奖励 > 8.5）：
- 第一摆杆角度：< 3°
- 第二摆杆角度：< 8°
- 角速度：接近 0
- 小车位置：< 1m

**良好策略**（奖励 6-8）：
- 第一摆杆角度：< 8°
- 第二摆杆角度：< 15°
- 角速度：< 2 rad/s
- 小车位置：< 2m

**可用策略**（奖励 4-6）：
- 第一摆杆角度：< 12°
- 第二摆杆角度：< 20°
- 偶尔失控但能恢复

---

## 🎬 录制演示视频

### 基础录制

```bash
python play.py --task double_pendulum \
    --model runs/double_pendulum_xxx/best_model/best_model.zip \
    --record \
    --episodes 3
```

### 高质量录制

```bash
python play.py --task double_pendulum \
    --model runs/double_pendulum_xxx/best_model/best_model.zip \
    --record \
    --episodes 5 \
    --output-dir videos/double_pendulum_demo
```

### 扰动测试录制

```bash
python play.py --task double_pendulum \
    --model runs/double_pendulum_xxx/best_model/best_model.zip \
    --record \
    --episodes 3 \
    --disturbance \
    --disturbance-force 3.0 \
    --disturbance-prob 0.015
```

视频将保存在 `videos/` 目录下。

---

## 📚 相关资源

- 📖 [奖励函数设计详解](reward_function_design.md)
- 📊 [奖励函数对比测试](../compare_reward_functions.py)
- 🔧 [配置文件](../configs/double_pendulum_config.py)
- 🎮 [环境实现](../envs/double_pendulum/double_pendulum_env.py)

---

**祝你训练顺利！** 🚀

如有问题，欢迎提 Issue 或查看文档。
