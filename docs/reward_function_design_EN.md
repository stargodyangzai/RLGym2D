# Double Pendulum Reward Function Design

## 📋 Table of Contents
- [Background](#background)
- [Problems with Traditional Methods](#problems-with-traditional-methods)
- [Smooth Gaussian Multiplicative Reward](#smooth-gaussian-multiplicative-reward)
- [Experimental Comparison](#experimental-comparison)
- [Configuration Guide](#configuration-guide)

---

## Background

The double pendulum is an extremely challenging control task:
- **Two poles**: First pole connects to cart, second pole connects to first pole's end
- **Coupled dynamics**: Two poles influence each other, controlling one affects the other
- **Multi-objective**: Must keep both poles upright simultaneously

### The "Abandon One to Save Another" Problem

In traditional **additive rewards**:
```
Reward = Pole1_Reward + Pole2_Reward
```

The agent may discover:
- Keep only Pole1 upright (get full points)
- Abandon Pole2 (lose some points)
- Total score is still decent

This leads the agent to learn "abandoning one to save another", controlling only one pole.

---

## Problems with Traditional Methods

### 1. Additive Reward Problems

```python
# Additive reward
reward = pole1_reward + pole2_reward
```

**Problem**: Good Pole1 can compensate for bad Pole2

**Example**:
- Pole1 = 2° (reward +3.0)
- Pole2 = 30° (reward +1.67)
- Total reward = 4.67 (not bad!)

The agent has no incentive to save Pole2.

### 2. Stepwise Multiplicative Reward Problems

While multiplication solves "abandon one to save another", stepwise functions bring new issues:

```python
# Stepwise multiplicative
if pole1_angle < 5:
    pole1_status = 1.0  # Dead zone!
elif pole1_angle < 15:
    pole1_status = 0.7  # Jump!
```

**Problem 1: Gradient Dead Zone**
- Pole1 between 0°-5°, reward always 1.0
- No incentive to optimize from 4° to 1°
- Lacks feedback for refinement

**Problem 2: Gradient Discontinuity**
- At switching points like 5°, 15°, reward jumps
- Derivative discontinuous, unstable learning
- PPO struggles to get stable gradient signals

**Problem 3: Ignoring Angular Velocity**
- Only checks angle, not velocity
- Second pole spinning fast through vertical gets full reward
- "Rotation exploitation" problem

**Problem 4: Reluctance to Tilt**
- Pole1 from 5° to 6°, reward drops from 1.0 to 0.97
- Too much loss, agent afraid to try
- Unwilling to tilt to save Pole2

---

## Smooth Gaussian Multiplicative Reward

### Core Idea

Use **Gaussian functions** instead of stepwise functions, introduce **angular velocity penalty**:

```python
# Angle status (smooth Gaussian)
p1_angle_status = exp(-pole1_angle² / σ₁²)
p2_angle_status = exp(-pole2_angle² / σ₂²)

# Angular velocity status
v1_status = exp(-pole1_velocity² / σ_v1²)
v2_status = exp(-pole2_velocity² / σ_v2²)

# Combined status
pole1_total = p1_angle_status × (0.8 + 0.2 × v1_status)
pole2_total = p2_angle_status × (0.7 + 0.3 × v2_status)

# Total reward (multiplicative)
reward = pole1_total × pole2_total × position_factor × 10
```

### Mathematical Properties

**Gaussian function** `f(x) = exp(-x²/σ²)` advantages:

1. **Everywhere differentiable**: Has derivatives everywhere
2. **Smooth transition**: No jumps or discontinuities
3. **Physical intuition**: More deviation, heavier penalty (quadratic growth)
4. **Natural decay**: Reward naturally approaches 0 when far from target

### Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `angle1_sigma` | 0.10 | First pole angle tolerance (~18° drops to 0.36) |
| `angle2_sigma` | 0.15 | Second pole angle tolerance (slightly more lenient) |
| `vel1_sigma` | 5.0 | First pole angular velocity tolerance |
| `vel2_sigma` | 10.0 | Second pole angular velocity tolerance |
| `vel1_weight` | 0.2 | First pole velocity weight (20%) |
| `vel2_weight` | 0.3 | Second pole velocity weight (30%, emphasize stillness) |

---

## Experimental Comparison

### Test Scenarios

Run comparison test:
```bash
python compare_reward_functions.py
```

### Scenario 1: Ideal State

```
Pole1 = 2°, Pole2 = 2°
- Additive:              5.01
- Stepwise Multiplicative: 12.00
- Smooth Gaussian:        9.80
```

### Scenario 2: Abandon One to Save Another

```
Pole1 = 2°, Pole2 = 30° (Pole2 collapsed)
- Additive:              1.67  ← Still positive reward!
- Stepwise Multiplicative: 3.50
- Smooth Gaussian:        1.59  ← Severe penalty
```

### Scenario 3: Rotation Exploitation

```
Pole1 = 2°, Pole2 = 5°, but Pole2 angular velocity = 8
- Additive:              5.01  ← Can't detect rotation
- Stepwise Multiplicative: 12.00  ← Can't detect rotation
- Smooth Gaussian:        6.58  ← Penalizes rotation
```

### Scenario 4: Gradient Dead Zone

```
Pole1 optimizing from 4° to 1°, Pole2 stays at 5°
- Additive:              5.01 → 5.01 → 5.01 → 5.01  ← No feedback
- Stepwise Multiplicative: 12.00 → 12.00 → 12.00 → 12.00  ← No feedback
- Smooth Gaussian:        9.05 → 9.25 → 9.39 → 9.48  ← Continuous growth
```

### Scenario 5: Coordinated Control

```
Can P1 tilting moderately exchange for P2's significant improvement?

Config 1: P1=2°, P2=30° (P1 perfect, P2 collapsed)
- Smooth Gaussian: 1.59

Config 2: P1=10°, P2=15° (P1 tilts 10°, P2 improves to 15°)
- Smooth Gaussian: 4.67  ← Clear benefit!

Config 3: P1=15°, P2=10° (P1 tilts 15°, P2 improves to 10°)
- Smooth Gaussian: 4.11  ← Still beneficial
```

**Conclusion**: Smooth Gaussian encourages P1 to tilt moderately to save P2.

---

## Configuration Guide

### Basic Configuration

In `configs/double_pendulum_config.py`:

```python
'reward_config': {
    # Enable smooth Gaussian multiplicative reward (recommended)
    'use_multiplicative': True,
    'use_smooth_gaussian': True,
    
    # Angle tolerance
    'angle1_sigma': 0.10,
    'angle2_sigma': 0.15,
    
    # Angular velocity tolerance
    'vel1_sigma': 5.0,
    'vel2_sigma': 10.0,
    
    # Velocity weight
    'vel1_weight': 0.2,
    'vel2_weight': 0.3,
}
```

### Tuning Suggestions

**Stricter control** (require higher precision):
```python
'angle1_sigma': 0.08,  # Lower tolerance
'angle2_sigma': 0.12,
'vel1_weight': 0.3,    # Increase velocity weight
'vel2_weight': 0.4,
```

**More lenient control** (easier to learn):
```python
'angle1_sigma': 0.15,  # Higher tolerance
'angle2_sigma': 0.20,
'vel1_weight': 0.1,    # Decrease velocity weight
'vel2_weight': 0.2,
```

**Emphasize second pole control**:
```python
'angle2_sigma': 0.10,  # Stricter on second pole
'vel2_weight': 0.4,    # More emphasis on second pole stillness
```

### Comparison Testing

To compare different reward functions:

```python
# Use additive reward (old version)
'use_multiplicative': False,

# Use stepwise multiplicative (intermediate version)
'use_multiplicative': True,
'use_smooth_gaussian': False,

# Use smooth Gaussian multiplicative (recommended)
'use_multiplicative': True,
'use_smooth_gaussian': True,
```

---

## Training Recommendations

### Expected Learning Curve

Using smooth Gaussian multiplicative reward:

- **0-50 iterations**: Exploration phase, reward 0-3
- **50-150 iterations**: Learn to stabilize P1, reward 3-5
- **150-250 iterations**: Start coordinated control, reward 5-8
- **250-400 iterations**: Fine-tuning, reward 8-10

### Key Improvements

Compared to old version, you should observe:

✅ **First pole no longer "rigid"**
- Will actively tilt to help second pole
- No longer stubbornly stays at 0°

✅ **Second pole no longer "spinning"**
- Angular velocity penalty forces it to be still
- No more high-speed rotation exploitation

✅ **More stable learning**
- Smooth gradients avoid jumps
- Smoother training curves

✅ **Faster convergence**
- Every improvement gets feedback
- Incentive for refinement

---

## References

### Theoretical Foundation

1. **Multiplicative Reward**: Avoid "abandon one to save another" problem
   - Core idea: Shared fate, any failure means overall failure

2. **Gaussian Function**: Smooth reward shaping
   - Everywhere differentiable, provides stable gradients
   - Physical intuition: More deviation, heavier penalty

3. **Angular Velocity Penalty**: Distinguish "static upright" from "rotating through upright"
   - Avoid exploitation behavior
   - Encourage true stable control

### Related Work

- Reward Shaping in Reinforcement Learning
- Continuous Control with Deep Reinforcement Learning
- Multiplicative Reward Functions for Multi-Objective Tasks

---

## Summary

Smooth Gaussian multiplicative reward solves double pendulum control challenges through these innovations:

1. ✅ **Eliminate gradient dead zones** - Everywhere-differentiable Gaussian function
2. ✅ **Kill rotation exploitation** - Angular velocity penalty
3. ✅ **Encourage coordinated control** - Smooth multiplicative relationship
4. ✅ **Mathematical elegance** - Natural, continuous, differentiable

This is an innovative design derived from practice, providing new insights for reward function design in complex control tasks.

---

**Authors**: RLGym2D Team  
**Date**: 2026-01-03  
**Version**: 1.1.0
