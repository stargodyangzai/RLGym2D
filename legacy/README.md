# Legacy Files（旧版本文件）

这个目录包含项目重构前的旧版本文件，仅用于向后兼容。

## ⚠️ 注意

**新项目请使用根目录的文件：**
- 训练：`python train.py --task arm`
- 演示：`python play.py --task arm --model xxx`

**这些旧文件仅供参考和兼容：**
- `train_arm.py` - 旧的机械臂训练脚本
- `play_arm.py` - 旧的机械臂演示脚本
- `config.py` - 旧的配置文件（现在是兼容层）
- `simple_arm_env.py` - 旧的环境文件（现在是兼容层）
- `reward_functions.py` - 旧的奖励函数（现在是兼容层）

## 迁移指南

### 旧方式（legacy）
```bash
python legacy/train_arm.py --envs 32
python legacy/play_arm.py --model runs/xxx/model
```

### 新方式（推荐）
```bash
python train.py --task arm --envs 32
python play.py --task arm --model runs/xxx/model
```

## 何时删除这些文件？

当你确认新版本完全稳定后，可以删除 `legacy/` 目录。

建议保留至少一个月，确保没有依赖问题。
