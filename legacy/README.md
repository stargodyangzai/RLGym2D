# Legacy Files | 向后兼容文件

This directory contains deprecated files for backward compatibility.
这个目录包含用于向后兼容的已弃用文件。

## ⚠️ Deprecation Notice | 弃用通知

**These files are deprecated and will be removed in future versions.**
**这些文件已弃用，将在未来版本中移除。**

For new projects, please use the modern API:
新项目请使用现代API：

```python
# ❌ Old way (deprecated) | 旧方式（已弃用）
from legacy.simple_arm_env import SimpleArmEnv
from legacy.reward_functions import RewardFunction
from legacy.config import get_config

# ✅ New way (recommended) | 新方式（推荐）
from envs.arm import ArmEnv
from core.base_rewards import RewardFunction
from configs import TASK_CONFIGS
```

## 📁 File Descriptions | 文件说明

### Compatibility Layers | 兼容层
- **`simple_arm_env.py`** - Robotic arm environment compatibility layer | 机械臂环境兼容层
- **`reward_functions.py`** - Reward functions compatibility layer | 奖励函数兼容层
- **`config.py`** - Configuration compatibility layer | 配置兼容层

### Legacy Scripts | 旧版脚本
- **`train_arm.py`** - Old arm training script | 旧版机械臂训练脚本
- **`play_arm.py`** - Old arm demo script | 旧版机械臂演示脚本

## 🔄 Migration Guide | 迁移指南

### Training Scripts | 训练脚本
```bash
# Old way | 旧方式
python legacy/train_arm.py --envs 32

# New way | 新方式
python train.py --task arm --envs 32
```

### Demo Scripts | 演示脚本
```bash
# Old way | 旧方式
python legacy/play_arm.py --model model.zip

# New way | 新方式
python play.py --task arm --model model.zip
```

### Environment Usage | 环境使用
```python
# Old way | 旧方式
from legacy.simple_arm_env import SimpleArmEnv
env = SimpleArmEnv()

# New way | 新方式
from envs.arm import ArmEnv
env = ArmEnv()
```

### Configuration | 配置
```python
# Old way | 旧方式
from legacy.config import get_config
config = get_config()

# New way | 新方式
from configs.arm_config import ARM_CONFIG
config = ARM_CONFIG
```

## 📅 Removal Timeline | 移除时间表

- **v1.0.0** - Files moved to legacy/ | 文件移至legacy/
- **v1.5.0** - Deprecation warnings added | 添加弃用警告
- **v2.0.0** - Files will be removed | 文件将被移除

## 🆘 Need Help? | 需要帮助？

If you encounter issues migrating from legacy files:
如果在从旧文件迁移时遇到问题：

1. Check the [Migration Guide](../README.md#migration) | 查看[迁移指南](../README.md#migration)
2. Open an [Issue](https://github.com/stargodyangzai/RLGym2D/issues) | 提交[Issue](https://github.com/stargodyangzai/RLGym2D/issues)
3. Reference the [Documentation](../README.md) | 参考[文档](../README.md)

---

**Note**: These files are provided for backward compatibility only. New features and bug fixes will not be applied to legacy files.

**注意**: 这些文件仅用于向后兼容。新功能和错误修复不会应用于旧文件。
