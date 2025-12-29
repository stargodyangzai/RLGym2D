"""
测试火柴人的终止条件和成功判定
"""
import numpy as np
from envs.walker.walker_env import WalkerEnv
from configs.walker_config import WALKER_CONFIG

# 创建环境
env = WalkerEnv(config=WALKER_CONFIG['env_config'])

print("=" * 60)
print("测试火柴人环境")
print("=" * 60)

# 测试10个episode
for episode in range(10):
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done:
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        done = terminated or truncated
        
        # 如果终止，打印原因
        if terminated:
            print(f"\nEpisode {episode + 1}:")
            print(f"  步数: {step_count}")
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  距离: {info['distance']:.2f}m")
            print(f"  速度: {info['velocity']:.2f}m/s")
            print(f"  躯干高度: {env.torso.position.y:.1f}")
            print(f"  躯干角度: {env.torso.angle:.2f} rad ({np.degrees(env.torso.angle):.1f}°)")
            print(f"  成功: {info['success']}")
            
            # 判断终止原因
            if env.torso.position.y > 520:
                print(f"  终止原因: 躯干高度太低 (y={env.torso.position.y:.1f} > 520)")
            elif abs(env.torso.angle) > np.pi / 3:
                print(f"  终止原因: 躯干倾斜太大 (|angle|={abs(env.torso.angle):.2f} > {np.pi/3:.2f})")
            break
        
        if truncated:
            print(f"\nEpisode {episode + 1}:")
            print(f"  步数: {step_count}")
            print(f"  总奖励: {total_reward:.2f}")
            print(f"  距离: {info['distance']:.2f}m")
            print(f"  速度: {info['velocity']:.2f}m/s")
            print(f"  成功: {info['success']}")
            print(f"  终止原因: 达到最大步数")
            break

env.close()
print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
