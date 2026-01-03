# RLGym2D Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-01-03

### Added
- 🎯 **Double Pendulum Task** - Extremely challenging control task
- 🧮 **Smooth Gaussian Multiplicative Reward** - Innovative reward design solving the "abandon one to save another" problem

### Improved
- ✨ **Major Reward Function Improvement** (Double Pendulum)
  - Upgraded from stepwise multiplicative to smooth Gaussian multiplicative reward
  - Using `exp(-x²/σ²)` instead of hard-segmented functions, eliminating gradient dead zones
  - Introduced angular velocity penalty, killing the "rotation exploitation" problem
  - Smooth multiplicative relationship, encouraging coordinated control of both poles
  - Everywhere-differentiable reward function, providing stable learning gradients

### Technical Details
- **Problem 1: Gradient Dead Zone**
  - Old: Reward constant at 1.0 when pole < 5°, no incentive for refinement
  - New: Gaussian function has derivatives everywhere, every improvement gets feedback
  
- **Problem 2: Rotation Exploitation**
  - Old: Only checks angle, high-speed rotation through vertical gets full reward
  - New: Angular velocity penalty, only "static upright" gets high reward
  
- **Problem 3: Reluctance to Tilt**
  - Old: P1 from 5° to 6° drops reward from 1.0 to 0.97, too much loss
  - New: Smooth transition, P1 tilting moderately to save P2 shows clear benefit
  
- **Problem 4: Gradient Discontinuity**
  - Old: Stepwise function has discontinuous derivatives at switching points, unstable learning
  - New: Gaussian function is everywhere differentiable, PPO gets stable gradient signals

### Configuration Options
```python
'reward_config': {
    'use_multiplicative': True,   # Enable multiplicative reward
    'use_smooth_gaussian': True,  # Enable smooth Gaussian (recommended)
    'angle1_sigma': 0.10,         # First pole angle tolerance
    'angle2_sigma': 0.15,         # Second pole angle tolerance
    'vel1_sigma': 5.0,            # First pole angular velocity tolerance
    'vel2_sigma': 10.0,           # Second pole angular velocity tolerance
    'vel1_weight': 0.2,           # First pole velocity weight
    'vel2_weight': 0.3,           # Second pole velocity weight
}
```

### Tools
- 📊 `compare_reward_functions.py` - Test script comparing three reward functions
  - Additive vs Stepwise Multiplicative vs Smooth Gaussian Multiplicative
  - Gradient test, rotation exploitation test, coordination control test
  - Numerical analysis and visualization comparison

## [1.0.0] - 2024-12-29

### Added
- 🎯 CartPole balance control task
- 🌪️ Multi-type disturbance testing functionality
- 🤖 Robotic arm target reaching task
- 🚶 Humanoid walking task
- ⚙️ Unified configuration system
- 📊 W&B experiment tracking support
- 🎨 Beautiful visualization interface
- 📈 TensorBoard logging
- 🔧 Modular reward system

### Features
- GPU/CPU auto-detection support
- Parallel training environments
- Real-time performance monitoring
- Automatic model saving
- Backward compatibility design

### Tech Stack
- Stable-Baselines3 (PPO algorithm)
- Gymnasium (environment interface)
- PyMunk (physics simulation)
- Pygame (visualization)
- PyTorch (deep learning)

## [Future Plans]

### Coming Soon
- [ ] 3D robotic arm task
- [ ] Obstacle avoidance
- [ ] Curriculum learning
- [ ] More algorithm support

### Long-term
- [ ] Visual input support
- [ ] Multi-agent cooperation
- [ ] Real robot interface
- [ ] Sim-to-real transfer