"""
测试火柴人环境
"""
from envs.walker.walker_env import WalkerEnv
import numpy as np

print("创建火柴人环境...")
env = WalkerEnv(render_mode='human')

print("环境信息:")
print(f"  观察空间: {env.observation_space}")
print(f"  动作空间: {env.action_space}")

print("\n开始测试...")
obs, info = env.reset()
print(f"初始观察: {obs.shape}")

for step in range(500):
    # 随机动作
    action = env.action_space.sample()
    
    # 执行
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 渲染
    env.render()
    
    if terminated or truncated:
        print(f"\n回合结束 @ 步数 {step}")
        print(f"  原因: {'摔倒' if terminated else '超时'}")
        print(f"  距离: {info.get('distance', 0):.2f}")
        break

env.close()
print("\n测试完成！")
