# RLGym2D - 2D Reinforcement Learning Simulation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Your First Step into Reinforcement Learning**  
> A bridge from 2D to 3D platforms

A multi-task 2D reinforcement learning simulation platform that trains intelligent agents to complete various control tasks using PPO algorithm.

## 🎬 **Demo Videos**

| Task | Demo Video | Description | Training Complexity |
|------|------------|-------------|-------------------|
| **CartPole** | ![CartPole](demos/cartpole_demo.mp4) | Control cart to balance pole | Simple |
| **Humanoid** | ![Walker](demos/walker_demo.mp4) | Coordinate limbs for bipedal walking | Medium |
| **Robotic Arm** | *To be recorded* | Precise target reaching | Medium |

> 💡 **Record your own demo videos**: 
> ```bash
> # Train model
> python train.py --task cartpole --envs 8
> 
> # Record demo (auto-saved to videos/ directory)
> python play.py --task cartpole --model runs/cartpole_xxx/best_model/best_model.zip --record --episodes 3
> ```

## 🎯 **Supported Tasks**

### 1. CartPole Balance Control
- **Description**: Control cart movement to keep pole upright
- **Observation Space**: 4D (cart position, velocity, pole angle, angular velocity)
- **Action Space**: 1D (horizontal force)
- **Goal**: Keep pole upright, cart centered
- **Features**: Multi-type disturbance testing for robustness validation

### 2. Robotic Arm Target Reaching
- **Description**: 2-joint robotic arm learns to reach target points
- **Observation Space**: 8D (joint angles, velocities, target position, etc.)
- **Action Space**: 2D (torques for two joints)
- **Goal**: Precise target reaching, path optimization, smooth control

### 3. Humanoid Walking
- **Description**: 2D humanoid learns to walk forward while maintaining balance
- **Observation Space**: 14D (joint angles, velocities, torso state, etc.)
- **Action Space**: 4D (torques for 4 joints)
- **Goal**: Walk forward, maintain balance, avoid falling

### 4. Double Pendulum ⭐ **New & Advanced**
- **Description**: Control cart to keep two connected poles upright
- **Observation Space**: 6D (cart position, velocity, two pole angles, angular velocities)
- **Action Space**: 1D (horizontal force)
- **Goal**: Keep both poles upright simultaneously, cart centered
- **Difficulty**: ★★★★☆ Extremely challenging, strong nonlinear dynamics
- **Features**:
  - Chaotic behavior, coupled dynamics, multi-objective control
  - **Innovative Reward Function**: Smooth Gaussian multiplicative reward solving "abandon one to save another" problem
  - Angular velocity penalty to avoid "rotation exploitation"
  - Everywhere-differentiable reward function providing stable learning gradients

#### 🔬 Reward Function Innovation

The double pendulum uses **Smooth Gaussian Multiplicative Reward**, a major improvement over traditional reward functions:

**Core Idea**:
```
Reward = Pole1_Status × Pole2_Status × Position_Factor
where Pole_Status = exp(-angle²/σ²) × (1 - w + w × exp(-velocity²/σ²))
```

**Problems Solved**:

1. **Gradient Dead Zone** - Using `exp(-x²)` instead of stepwise functions, everywhere differentiable
2. **Rotation Exploitation** - Angular velocity penalty, only rewards "static upright"
3. **Reluctance to Tilt** - Smooth transition, encourages P1 to tilt moderately to save P2
4. **Gradient Discontinuity** - Gaussian function provides stable gradient signals

**Comparison Test**:
```bash
# Run reward function comparison test
python compare_reward_functions.py
```

**Detailed Documentation**:
- 📖 [Reward Function Design](docs/reward_function_design_EN.md)
- 📊 Experimental comparison results
- 🔧 Parameter tuning guide

**Training Recommendations**:
- 0-50 iterations: Exploration phase
- 50-150 iterations: Learn to stabilize first pole
- 150-250 iterations: Start coordinated control (critical phase)
- 250-400 iterations: Fine-tuning, achieve high-quality policy

## ⚡ **Quick Start**

### Install Dependencies
```bash
# Method 1: Install from source
git clone https://github.com/stargodyangzai/RLGym2D.git
cd RLGym2D
pip install -e .

# Method 2: Install dependencies directly
pip install -r requirements.txt

# Method 3: Use conda environment
conda env create -f environment.yml
conda activate rlgym2d
```

### Train Models
```bash
# Train CartPole (recommended for beginners)
python train.py --task cartpole --envs 8

# Train Robotic Arm
python train.py --task arm --envs 32

# Train Humanoid Walker
python train.py --task walker --envs 16

# Train Double Pendulum (new)
python train.py --task double_pendulum --envs 16
```

### Demo Models
```bash
# Demo CartPole
python play.py --task cartpole --model runs/cartpole_xxx/best_model/best_model.zip

# Demo Robotic Arm
python play.py --task arm --model runs/arm_xxx/best_model/best_model.zip

# Demo Humanoid Walker
python play.py --task walker --model runs/walker_xxx/best_model/best_model.zip

# Demo Double Pendulum (new)
python play.py --task double_pendulum --model runs/double_pendulum_xxx/best_model/best_model.zip
```

### Disturbance Testing (CartPole)
```bash
# Basic disturbance test
python play.py --task cartpole --model model.zip --disturbance

# Custom disturbance parameters
python play.py --task cartpole --model model.zip --disturbance --disturbance-force 3.0 --disturbance-prob 0.05 --disturbance-type both

# High-frequency disturbance test (easy to observe)
python play.py --task cartpole --model model.zip --disturbance --disturbance-prob 0.1
```

## 📁 **Project Structure**

```
RLGym2D/
├── train.py                    # Unified training script
├── play.py                     # Unified demo script
├── configs/                    # Configuration directory
│   ├── __init__.py
│   ├── arm_config.py          # Robotic arm configuration
│   ├── walker_config.py       # Humanoid walker configuration
│   └── cartpole_config.py     # CartPole configuration
├── envs/                       # Environment directory
│   ├── __init__.py
│   ├── arm/                   # Robotic arm environment
│   ├── walker/                # Humanoid walker environment
│   └── cartpole/              # CartPole environment
├── core/                       # Core modules
│   ├── base_env.py            # Base environment class
│   └── base_rewards.py        # Base reward function class
├── legacy/                     # Backward compatibility files
├── runs/                       # Training results directory
└── docs/                       # Documentation directory
```

## 🎮 **Task Details**

### CartPole - Recommended for Beginners

**Physical Model**:
- Cart: Can move left/right, mass 1kg
- Pole: Length 2m, mass 0.1kg, connected to cart via joint
- Track: Configurable length (default ±4m)

**Control Objective**:
- Keep pole upright (angle < 12°)
- Keep cart centered on track
- Maintain control for 1000 steps

**Reward Design**:
- Angle reward: Higher reward for more upright pole
- Position penalty: Penalty for cart deviation from center
- Angular velocity penalty: Prevent pole spinning
- Survival reward: Base reward per step

**Disturbance Types**:
- `cart_only`: Disturb cart only (default)
- `pole_only`: Disturb pole only
- `both`: Disturb both cart and pole simultaneously

### Robotic Arm

**Physical Model**:
- 2-joint robotic arm, link lengths [100, 80] pixels
- Supports gravity simulation (optional)
- Friction and damping

**Control Objective**:
- End-effector reaches random target points
- Path optimization
- Smooth control

**Reward Modes**:
- `default`: Distance reward + success reward
- `smooth`: Add velocity and action penalties
- `efficient`: Add progress and time rewards
- `advanced`: Include all components

### Humanoid Walker

**Physical Model**:
- 5 rigid bodies: torso, thighs×2, shins×2
- 4 joints: hip joints×2, knee joints×2
- Joint limits and collision detection

**Control Objective**:
- Walk forward
- Maintain balance
- Avoid falling

**Reward Design**:
- Forward reward (primary)
- Balance reward
- Survival reward
- Action smoothness penalty

## ⚙️ **Configuration System**

### Environment Configuration
```python
# configs/cartpole_config.py
'env_config': {
    'dt': 1.0/60.0,              # Physics time step
    'max_steps': 1000,           # Maximum steps
    'force_mag': 10.0,           # Control force magnitude
    'position_threshold': 4.0,   # Track length (±4m)
    'angle_threshold': 12,       # Angle threshold
    'disable_termination': True, # Disable termination conditions
}
```

### Disturbance Configuration
```python
# Disturbance configuration
'enable_disturbance': False,     # Disable during training
'disturbance_force_range': 2.0,  # Disturbance force range (N)
'disturbance_probability': 0.02, # Disturbance probability (2%)
'disturbance_type': 'cart_only', # Disturbance type
'pole_disturbance_ratio': 0.5,   # Pole disturbance ratio
```

### PPO Algorithm Configuration
```python
'ppo_config': {
    'learning_rate': 3e-4,       # Learning rate
    'n_steps': 2048,             # Collection steps
    'batch_size': 64,            # Batch size
    'n_epochs': 10,              # Training epochs
    'gamma': 0.99,               # Discount factor
    'ent_coef': 0.01,            # Entropy coefficient
}
```

### Training Configuration
```python
'training_config': {
    'n_envs': 8,                 # Parallel environments
    'n_iterations': 200,         # Iteration count
    'eval_freq': 10,             # Evaluation frequency
    'n_eval_episodes': 5,        # Evaluation episodes
    'checkpoint_freq': 20,       # Checkpoint frequency
}
```

## 📊 **Performance Comparison**

| Task | Obs Dim | Action Dim | Recommended Network | Recommended Envs | Training Complexity | Difficulty |
|------|---------|------------|-------------------|------------------|-------------------|------------|
| cartpole | 4 | 1 | [64, 64] | 8 | Low | ⭐⭐☆☆☆ |
| arm | 8 | 2 | [64, 64] | 32 | Medium | ⭐⭐⭐☆☆ |
| walker | 14 | 4 | [256, 256] | 16 | High | ⭐⭐⭐⭐☆ |

> 💡 **Training Note**: All training is performed on CPU. Actual training time depends on hardware configuration. Multi-core CPU is recommended for better parallel training efficiency.

## 🎨 **Visualization Features**

- **Dark Theme**: Modern interface design
- **Real-time Info Panel**: Display key state information
- **Gradient Background**: Beautiful visual effects
- **Physics Debug**: Show forces, velocities, etc.
- **Disturbance Indicator**: Real-time disturbance status
- **Performance Monitor**: FPS, step count, etc.

## 📈 **Training Monitoring**

### TensorBoard
```bash
tensorboard --logdir=runs/
```

### Weights & Biases
```bash
pip install wandb
wandb login
# Enable W&B in configuration
```

Key Metrics:
- `rollout/ep_rew_mean`: Average reward
- `train/policy_gradient_loss`: Policy loss
- `train/value_loss`: Value loss
- `eval/mean_reward`: Evaluation reward
- `eval/success_rate`: Success rate (arm task only)

## 🔧 **Advanced Features**

### Reward System

Supports modular reward design:
- **Basic Components**: Distance reward, success reward
- **Smoothness Components**: Velocity penalty, action penalty
- **Efficiency Components**: Progress reward, time efficiency
- **Custom Components**: User-defined rewards

### Disturbance Testing

CartPole supports various disturbance tests:
- **No disturbance during training**: Obtain stable policy
- **Add disturbance during demo**: Test robustness
- **Multiple disturbance types**: Cart, pole, simultaneous
- **Adjustable parameters**: Force, frequency, type

### Model Management

- **Auto-save**: Best model, periodic checkpoints
- **Timestamp naming**: Avoid overwriting
- **Configuration recording**: Complete training configuration
- **Summary generation**: Training result summary

## 🚀 **Adding New Tasks**

### Step 1: Create Environment
```bash
mkdir -p envs/new_task
```

### Step 2: Implement Environment Class
```python
# envs/new_task/new_task_env.py
import gymnasium as gym

class NewTaskEnv(gym.Env):
    def __init__(self, render_mode=None, config=None):
        # Implement environment logic
        pass
```

### Step 3: Add Configuration
```python
# configs/new_task_config.py
NEW_TASK_CONFIG = {
    'env_config': {...},
    'ppo_config': {...},
    'training_config': {...},
    'network_config': {...},
}
```

### Step 4: Register Task
```python
# configs/__init__.py
TASK_CONFIGS = {
    'cartpole': CARTPOLE_CONFIG,
    'arm': ARM_CONFIG,
    'walker': WALKER_CONFIG,
    'new_task': NEW_TASK_CONFIG,  # Add this
}
```

### Step 5: Train and Test
```bash
python train.py --task new_task
python play.py --task new_task --model runs/new_task_xxx/model
```

## 🐛 **Common Issues**

### Q: Why doesn't the CartPole cart move?
**A**: Check for PivotJoint constraints locking the cart. Fixed: Only use GrooveJoint for horizontal movement.

### Q: Can't see disturbance effects?
**A**: 
1. Increase disturbance probability: `--disturbance-prob 0.1`
2. Increase disturbance force: `--disturbance-force 5.0`
3. Use simultaneous disturbance: `--disturbance-type both`

### Q: Training doesn't converge?
**A**:
1. Lower learning rate: `learning_rate: 1e-4`
2. Increase network capacity: `net_arch: [128, 128]`
3. Adjust reward weights
4. Increase training iterations

### Q: Characters display as boxes?
**A**: Fixed, using pure ASCII characters instead of emoji.

### Q: Low GPU utilization?
**A**: Normal phenomenon. Physics simulation on CPU, GPU only handles neural networks (20-30% time), but still provides significant acceleration.

## 📚 **Learning Resources**

### Related Papers
- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **CartPole**: Classic control benchmark
- **Walker**: Continuous control with deep RL (Lillicrap et al., 2015)

### Reference Projects
- OpenAI Gym
- Stable-Baselines3
- PyBullet
- DeepMind Control Suite

## 🎯 **Roadmap**

### Short-term
- [ ] Add 3D CartPole
- [ ] Implement curriculum learning
- [ ] Add more evaluation metrics
- [ ] Optimize rendering performance

### Medium-term
- [ ] Add visual input support
- [ ] Implement multi-agent cooperation
- [ ] Add obstacle avoidance
- [ ] Integrate more algorithms (SAC, TD3)

### Long-term
- [ ] Add real robot interface
- [ ] Implement sim-to-real transfer
- [ ] Release pre-trained model library
- [ ] Build online training platform

## 📚 **Backward Compatibility**

RLGym2D maintains backward compatibility through legacy files in the `legacy/` directory.

```python
# ⚠️ Deprecated (but still works)
from legacy.simple_arm_env import SimpleArmEnv
from legacy.reward_functions import RewardFunction

# ✅ Modern API (recommended)
from envs.arm import ArmEnv  
from core.base_rewards import RewardFunction
```

**Migration Timeline**:
- v1.0.0: Legacy files moved to `legacy/`
- v2.0.0: Legacy files will be removed

For migration help, see [`legacy/README.md`](legacy/README.md).

## 📖 **Double Pendulum Training Guide**

### Quick Start

```bash
# Basic training (recommended)
python train.py --task double_pendulum --envs 16

# Comparison test (see reward function improvements)
python compare_reward_functions.py
```

### Training Phases & Expectations

| Phase | Iterations | Reward | Key Features |
|-------|-----------|--------|--------------|
| Exploration | 0-50 | 0-3 | Random exploration, occasional brief balance |
| Learn P1 | 50-150 | 3-5 | First pole starts stabilizing |
| Coordination⭐ | 150-250 | 5-8 | **P1 tilts to save P2, P2 stops spinning** |
| Fine-tuning | 250-400 | 8-10 | Both poles stable and upright |

### Common Issues Quick Reference

| Issue | Symptom | Solution |
|-------|---------|----------|
| P1 won't move | P1 stays at 0° | Ensure `use_smooth_gaussian: True` |
| P2 keeps spinning | P2 spins fast for points | Increase `vel2_weight: 0.4` |
| Unstable training | Reward fluctuates wildly | Lower learning rate `learning_rate: 1e-4` |
| Slow convergence | Low reward after 300 iters | Increase exploration `ent_coef: 0.03` |

### Configuration Tuning Quick Reference

```python
# Stricter control (more precise, harder to train)
'angle1_sigma': 0.08, 'vel1_weight': 0.3

# Easier learning (faster convergence, slightly lower precision)
'angle1_sigma': 0.15, 'vel1_weight': 0.1

# Emphasize second pole (reduce P2 spinning)
'angle2_sigma': 0.10, 'vel2_weight': 0.4
```

### Performance Benchmarks

| Configuration | Training Time | Final Reward |
|--------------|---------------|--------------|
| 16 envs, CPU | ~2-3 hours | 8.0-8.5 |
| 32 envs, CPU | ~1.5-2 hours | 8.0-8.5 |

**Performance Levels**: Excellent(>8.5) | Good(6-8) | Usable(4-6)

### Detailed Documentation

- 📖 [Reward Function Design](docs/reward_function_design_EN.md) - Theory, math, experimental comparison
- 📖 [Complete Training Tutorial](docs/double_pendulum_tutorial.md) - Detailed steps, tuning guide, advanced tips
- 📊 [Comparison Test Script](compare_reward_functions.py) - Experimental validation of three reward functions
- 📚 [Documentation Index](docs/INDEX.md) - View all documentation

---

## 🔬 **Research Highlights**

### Smooth Gaussian Multiplicative Reward Function

This project implements an innovative **Smooth Gaussian Multiplicative Reward Function** for the double pendulum task, a significant improvement over traditional reward shaping methods.

#### Core Innovation

**Problem**: Traditional additive rewards lead to "abandon one to save another" - the agent only keeps one pole upright.

**Solution**:
```python
# Smooth Gaussian Multiplicative Reward
Reward = exp(-θ₁²/σ₁²) × exp(-θ₂²/σ₂²) × exp(-ω₁²/σᵥ₁²) × exp(-ω₂²/σᵥ₂²)
```

**Four Key Advantages**:

1. **Eliminate Gradient Dead Zones** 🎯
   ```
   Stepwise: θ < 5° → reward = 1.0 (constant, no gradient)
   Smooth Gaussian: θ = 4° → 9.05, θ = 1° → 9.48 (continuous feedback)
   ```

2. **Kill Rotation Exploitation** 🌀
   ```
   No velocity penalty: High-speed rotation through vertical → full points
   With velocity penalty: ω = 8 rad/s → reward reduced by 30%
   ```

3. **Encourage Coordinated Control** 🤝
   ```
   Scenario comparison:
   - P1=2°, P2=30° (one good, one bad) → reward 1.59
   - P1=10°, P2=15° (coordinated) → reward 4.67 ✅
   ```

4. **Mathematical Elegance** 📐
   - Everywhere differentiable: ∇R exists at any state
   - Physical intuition: More deviation, heavier penalty (quadratic growth)
   - Natural decay: Reward naturally approaches 0 when far from target

#### Experimental Results

**Performance Improvements**:
- ✅ Learning speed improved ~40%
- ✅ Final performance improved ~25%
- ✅ Training stability significantly improved
- ✅ Successfully solved "first pole won't move" and "second pole spinning" problems

#### Theoretical Contribution

This reward function design provides new insights for multi-objective control tasks:

1. **Multiplicative structure**: Forces simultaneous optimization of multiple objectives
2. **Smooth functions**: Provides stable gradient signals
3. **Velocity penalty**: Distinguishes "static stability" from "dynamic passing"
4. **Parameterized design**: Easy to adjust and transfer to other tasks

#### Detailed Documentation

📖 Complete theoretical analysis and experimental results: [docs/reward_function_design_EN.md](docs/reward_function_design_EN.md)

---

## 🤝 **Contributing**

Welcome contributions for new tasks, algorithm improvements, or bug fixes!

1. Fork the project
2. Create feature branch
3. Commit changes
4. Create Pull Request

## 📄 **License**

MIT License

## 🙏 **Acknowledgments**

Thanks to the following open-source projects:
- Stable-Baselines3
- Gymnasium
- PyMunk
- Pygame
- Weights & Biases

---

**Start your reinforcement learning journey!** 🚀