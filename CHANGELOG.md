# RLGym2D 更新日志

所有重要的项目更改都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2024-12-29

### 新增
- 🎯 倒立摆平衡任务（CartPole）
- 🌪️ 多种扰动测试功能
- 🤖 机械臂到达目标任务
- 🚶 火柴人行走任务
- ⚙️ 统一的配置系统
- 📊 W&B实验跟踪支持
- 🎨 美化的可视化界面
- 📈 TensorBoard日志记录
- 🔧 模块化奖励系统

### 特性
- 支持GPU/CPU自动检测
- 并行训练环境
- 实时性能监控
- 模型自动保存
- 向后兼容设计

### 技术栈
- Stable-Baselines3 (PPO算法)
- Gymnasium (环境接口)
- PyMunk (物理仿真)
- Pygame (可视化)
- PyTorch (深度学习)

## [未来计划]

### 即将发布
- [ ] 3D机械臂任务
- [ ] 障碍物避障
- [ ] 课程学习
- [ ] 更多算法支持

### 长期规划
- [ ] 视觉输入支持
- [ ] 多智能体协作
- [ ] 真实机器人接口
- [ ] Sim-to-real迁移