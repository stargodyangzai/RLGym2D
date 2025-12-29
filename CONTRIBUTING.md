# 贡献指南

感谢你对RLGym2D的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题
- 使用GitHub Issues报告bug
- 提供详细的复现步骤
- 包含系统信息和错误日志

### 提出新功能
- 先在Issues中讨论想法
- 说明功能的用途和价值
- 考虑向后兼容性

### 提交代码
1. Fork项目
2. 创建特性分支：`git checkout -b feature/new-task`
3. 提交更改：`git commit -m 'Add new task'`
4. 推送分支：`git push origin feature/new-task`
5. 创建Pull Request

## 📝 代码规范

### Python代码风格
- 遵循PEP 8规范
- 使用有意义的变量名
- 添加详细的中文注释
- 函数和类需要docstring

### 新任务添加
1. 在`envs/`下创建任务目录
2. 实现环境类继承`gym.Env`
3. 在`configs/`中添加配置
4. 更新`__init__.py`注册任务
5. 添加测试脚本

### 提交信息
- 使用清晰的提交信息
- 格式：`类型: 简短描述`
- 例如：`feat: 添加3D机械臂任务`

## 🧪 测试

提交前请确保：
- [ ] 代码能正常运行
- [ ] 新功能有对应的测试
- [ ] 不破坏现有功能
- [ ] 更新相关文档

## 📚 开发环境

```bash
# 克隆项目
git clone https://github.com/your-username/RLGym2D.git
cd RLGym2D

# 安装依赖
pip install -r requirements.txt

# 测试安装
python train.py --task cartpole --envs 2
```

## 🎯 优先级

我们特别欢迎以下贡献：
- 新的强化学习任务
- 算法改进和优化
- 文档完善
- Bug修复
- 性能优化

## 📞 联系方式

如有问题，可以通过以下方式联系：
- GitHub Issues
- 项目讨论区

感谢你的贡献！🚀