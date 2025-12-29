"""
测试修复后的火柴人环境
"""
import numpy as np
from envs.walker.walker_env import WalkerEnv
from configs.walker_config import WALKER_CONFIG

# 创建环境
env = WalkerEnv(render_mode='human', config=WALKER_CONFIG['env_config'])

print("=" * 60)
print("测试修复后的火柴人环境")
print("=" * 60)
print(f"地面高度: 550")
print(f"髋部高度: 460")
print(f"理想躯干高度: 430")
print(f"终止条件: 躯干y>510 或 |angle|>90°")
print("=" * 60)

# 测试5个episode
for episode in range(5):
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    max_distance = 0
    
    print(f"\n开始 Episode {episode + 1}")
    print(f"初始躯干高度: {env.torso.position.y:.1f}")
    print(f"初始躯干角度: {np.degrees(env.torso.angle):.1f}°")
    
    while not done and step_count < 500:  # 最多500步
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        step_count += 1
        max_distance = max(max_distance, info['distance'])
        done = terminated or truncated
        
        # 每50步打印一次状态
        if step_count % 50 == 0:
            print(f"  步数{step_count}: 距离={info['distance']:.2f}m, "
                  f"高度={env.torso.position.y:.1f}, "
                  f"角度={np.degrees(env.torso.angle):.1f}°")
    
    print(f"\nEpisode {episode + 1} 结束:")
    print(f"  总步数: {step_count}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  最远距离: {max_distance:.2f}m")
    print(f"  最终距离: {info['distance']:.2f}m")
    print(f"  最终速度: {info['velocity']:.2f}m/s")
    print(f"  最终高度: {env.torso.position.y:.1f}")
    print(f"  最终角度: {np.degrees(env.torso.angle):.1f}°")
    print(f"  成功: {info['success']}")
    
    if terminated:
        if env.torso.position.y > 510:
            print(f"  终止原因: 躯干高度太低 (y={env.torso.position.y:.1f} > 510)")
        elif abs(env.torso.angle) > np.pi / 2:
            print(f"  终止原因: 躯干倾斜太大 (|angle|={abs(np.degrees(env.torso.angle)):.1f}° > 90°)")
    elif truncated:
        print(f"  终止原因: 达到最大步数")

env.close()
print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
