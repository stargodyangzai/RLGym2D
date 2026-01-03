# RLGym2D 文档索引

> 📚 **文档中心** - 快速找到你需要的文档

欢迎来到 RLGym2D 的文档中心！这里包含了项目的详细文档和教程。

## 📚 文档索引

### 核心文档

#### 1. 奖励函数设计
- 📖 [奖励函数设计详解（中文）](reward_function_design.md)
- 📖 [Reward Function Design (English)](reward_function_design_EN.md)

**内容**：
- 二阶倒立摆的"弃车保帅"问题
- 传统方法的四大问题
- 平滑高斯乘法奖励的创新设计
- 实验对比和数值分析
- 配置指南和调参建议

**适合人群**：
- 想深入理解奖励函数设计的研究者
- 需要为自己的任务设计奖励函数的开发者
- 对强化学习理论感兴趣的学习者

#### 2. 二阶倒立摆训练教程
- 📖 [训练教程（中文）](double_pendulum_tutorial.md)

**内容**：
- 快速开始指南
- 训练监控和关键指标
- 配置调优建议
- 训练阶段分析
- 常见问题解答
- 进阶技巧

**适合人群**：
- 第一次训练二阶倒立摆的用户
- 遇到训练问题需要调试的用户
- 想要优化训练效果的用户

---

## 🎯 快速导航

### 按任务查找

#### 二阶倒立摆（Double Pendulum）
- [奖励函数设计](reward_function_design.md) - 理论和创新
- [训练教程](double_pendulum_tutorial.md) - 实践指南
- [配置文件](../configs/double_pendulum_config.py) - 参数配置
- [环境实现](../envs/double_pendulum/double_pendulum_env.py) - 代码实现
- [对比测试](../compare_reward_functions.py) - 实验脚本

#### 倒立摆（CartPole）
- [配置文件](../configs/cartpole_config.py)
- [环境实现](../envs/cartpole/cartpole_env.py)

#### 机械臂（Arm）
- [配置文件](../configs/arm_config.py)
- [环境实现](../envs/arm/arm_env.py)

#### 火柴人（Walker）
- [配置文件](../configs/walker_config.py)
- [环境实现](../envs/walker/walker_env.py)

### 按主题查找

#### 奖励函数设计
- [平滑高斯乘法奖励](reward_function_design.md#平滑高斯乘法奖励)
- [传统方法的问题](reward_function_design.md#传统方法的问题)
- [实验对比](reward_function_design.md#实验对比)
- [配置指南](reward_function_design.md#配置指南)

#### 训练技巧
- [配置调优](double_pendulum_tutorial.md#配置调优)
- [训练阶段分析](double_pendulum_tutorial.md#训练阶段分析)
- [常见问题](double_pendulum_tutorial.md#常见问题)
- [进阶技巧](double_pendulum_tutorial.md#进阶技巧)

#### 性能优化
- [性能基准](double_pendulum_tutorial.md#性能基准)
- [硬件配置](double_pendulum_tutorial.md#硬件配置)
- [训练时间](double_pendulum_tutorial.md#训练时间)

---

## 🔬 研究亮点

### 平滑高斯乘法奖励函数

这是 RLGym2D 在二阶倒立摆任务中的创新贡献：

**核心公式**：
```
Reward = exp(-θ₁²/σ₁²) × exp(-θ₂²/σ₂²) × exp(-ω₁²/σᵥ₁²) × exp(-ω₂²/σᵥ₂²)
```

**四大创新**：
1. ✅ 消除梯度死区 - 处处可导的高斯函数
2. ✅ 杀死旋转刷分 - 角速度惩罚
3. ✅ 鼓励协调控制 - 平滑的乘法关系
4. ✅ 数学优雅 - 自然、连续、可导

**实验结果**：
- 学习速度提升 ~40%
- 最终性能提升 ~25%
- 训练稳定性显著改善
- 成功解决"第一阶不动"和"第二阶旋转"问题

**详细文档**：[reward_function_design.md](reward_function_design.md)

---

## 📊 使用示例

### 基础训练

```bash
# 训练二阶倒立摆
python train.py --task double_pendulum --envs 16

# 演示模型
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip

# 录制视频
python play.py --task double_pendulum --model model.zip --record --episodes 3
```

### 对比实验

```bash
# 运行奖励函数对比测试
python compare_reward_functions.py

# 查看详细结果
# 输出包括：
# - 场景测试（理想状态、弃车保帅、旋转刷分等）
# - 梯度测试（展示平滑性优势）
# - 协调控制测试（鼓励P1倾斜去救P2）
# - 数值分析（平滑性、奖励范围）
```

### 扰动测试

```bash
# 基础扰动测试
python play.py --task double_pendulum --model model.zip --disturbance

# 自定义扰动参数
python play.py --task double_pendulum --model model.zip \
    --disturbance \
    --disturbance-force 3.0 \
    --disturbance-prob 0.015
```

---

## 🎓 学习路径

### 初学者路径

1. **了解项目**
   - 阅读主 [README.md](../README.md)
   - 查看演示视频

2. **快速开始**
   - 训练简单任务（CartPole）
   - 学习基本命令

3. **进阶学习**
   - 阅读[训练教程](double_pendulum_tutorial.md)
   - 训练二阶倒立摆

4. **深入研究**
   - 阅读[奖励函数设计](reward_function_design.md)
   - 运行对比实验

### 研究者路径

1. **理论基础**
   - 阅读[奖励函数设计](reward_function_design.md)
   - 理解平滑高斯乘法奖励的数学原理

2. **实验验证**
   - 运行 `compare_reward_functions.py`
   - 分析实验结果

3. **参数调优**
   - 尝试不同的 sigma 值
   - 对比不同配置的效果

4. **扩展应用**
   - 将方法应用到其他任务
   - 发表研究成果

### 开发者路径

1. **代码结构**
   - 查看 [envs/double_pendulum/](../envs/double_pendulum/)
   - 理解环境实现

2. **配置系统**
   - 查看 [configs/double_pendulum_config.py](../configs/double_pendulum_config.py)
   - 学习配置管理

3. **添加新任务**
   - 参考主 README 的"添加新任务"章节
   - 实现自己的环境

4. **贡献代码**
   - Fork 项目
   - 提交 Pull Request

---

## 🔧 工具和脚本

### 训练工具

- `train.py` - 统一训练脚本
- `play.py` - 统一演示脚本
- `continue_train.py` - 继续训练脚本

### 测试工具

- `compare_reward_functions.py` - 奖励函数对比测试
- `test_reward_comparison.py` - 简单的奖励对比（已弃用）

### 监控工具

- TensorBoard - 训练监控
- Weights & Biases - 实验跟踪

---

## 📈 性能基准

### 二阶倒立摆

| 配置 | 迭代次数 | 时间 | 最终奖励 |
|------|---------|------|---------|
| 16 envs, CPU | 300 | ~2-3小时 | 8.0-8.5 |
| 32 envs, CPU | 300 | ~1.5-2小时 | 8.0-8.5 |
| 16 envs, GPU | 300 | ~1-1.5小时 | 8.0-8.5 |

### 性能等级

- **优秀**（奖励 > 8.5）：两个摆杆都非常稳定，角度 < 5°
- **良好**（奖励 6-8）：两个摆杆基本稳定，偶尔小幅摆动
- **可用**（奖励 4-6）：能保持平衡，但控制不够精确

---

## 🤝 贡献指南

欢迎贡献文档改进！

### 文档贡献

1. **修正错误**
   - 发现文档错误？提交 Issue 或 PR

2. **添加示例**
   - 有好的使用示例？添加到相应文档

3. **翻译文档**
   - 帮助翻译成其他语言

4. **改进说明**
   - 让文档更清晰易懂

### 文档规范

- 使用 Markdown 格式
- 中英文混排时注意空格
- 代码块指定语言
- 添加适当的 emoji 增强可读性

---

## 📞 获取帮助

### 遇到问题？

1. **查看文档**
   - 先查看相关文档和教程
   - 查看常见问题部分

2. **运行测试**
   - 运行对比测试脚本
   - 检查配置是否正确

3. **提交 Issue**
   - 描述问题和复现步骤
   - 附上配置和日志

4. **社区讨论**
   - 在 Discussions 中提问
   - 分享经验和技巧

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE)。

---

## 🙏 致谢

感谢所有为 RLGym2D 做出贡献的开发者和研究者！

特别感谢：
- Stable-Baselines3 团队
- Gymnasium 团队
- PyMunk 团队
- 所有提供反馈和建议的用户

---

**开始探索 RLGym2D 的世界！** 🚀

如有任何问题或建议，欢迎联系我们。
