# Contributing Guide

Thank you for your interest in RLGym2D! We welcome all forms of contributions.

## 🤝 How to Contribute

### Report Issues
- Use GitHub Issues to report bugs
- Provide detailed reproduction steps
- Include system information and error logs

### Suggest New Features
- Discuss ideas in Issues first
- Explain the purpose and value of the feature
- Consider backward compatibility

### Submit Code
1. Fork the project
2. Create feature branch: `git checkout -b feature/new-task`
3. Commit changes: `git commit -m 'Add new task'`
4. Push branch: `git push origin feature/new-task`
5. Create Pull Request

## 📝 Code Standards

### Python Code Style
- Follow PEP 8 standards
- Use meaningful variable names
- Add detailed comments (English preferred, Chinese acceptable)
- Functions and classes need docstrings

### Adding New Tasks
1. Create task directory under `envs/`
2. Implement environment class inheriting `gym.Env`
3. Add configuration in `configs/`
4. Update `__init__.py` to register task
5. Add test scripts

### Commit Messages
- Use clear commit messages
- Format: `type: brief description`
- Example: `feat: add 3D robotic arm task`

## 🧪 Testing

Before submitting, please ensure:
- [ ] Code runs normally
- [ ] New features have corresponding tests
- [ ] Doesn't break existing functionality
- [ ] Update related documentation

## 📚 Development Environment

```bash
# Clone project
git clone https://github.com/stargodyangzai/RLGym2D.git
cd RLGym2D

# Install dependencies
pip install -r requirements.txt

# Test installation
python train.py --task cartpole --envs 2
```

## 🎯 Priorities

We especially welcome contributions in:
- New reinforcement learning tasks
- Algorithm improvements and optimizations
- Documentation improvements
- Bug fixes
- Performance optimizations

## 📞 Contact

If you have questions, you can contact us through:
- GitHub Issues
- Project discussions

Thank you for your contribution! 🚀