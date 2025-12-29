"""
演示训练好的模型
"""
import os
import glob
from simple_arm_env import SimpleArmEnv
from stable_baselines3 import PPO


def find_latest_model():
    """查找最新的模型"""
    runs = sorted(glob.glob("runs/*/model.zip"), key=os.path.getmtime, reverse=True)
    if runs:
        return runs[0].replace('.zip', '')
    return None


def play_model(model_path=None, n_episodes=5, render=True):
    """演示模型"""
    print("=" * 70)
    print("模型演示")
    print("=" * 70)
    
    # 如果没有指定模型，查找最新的
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("\n错误：未找到训练好的模型")
            print("请先运行训练：python train.py")
            return
        print(f"\n使用最新模型: {model_path}")
    
    # 加载模型
    try:
        print(f"\n加载模型: {model_path}")
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"\n错误：模型文件不存在: {model_path}")
        print("\n可用的模型:")
        os.system("python list_models.py")
        return
    
    # 创建环境
    render_mode = "human" if render else None
    env = SimpleArmEnv(render_mode=render_mode)
    
    print(f"\n开始演示 ({n_episodes} 个回合)...")
    print("-" * 70)
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    # 运行多个回合
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"\n回合 {episode + 1}/{n_episodes}:")
        
        while True:
            # 使用训练好的策略
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step += 1
            
            if terminated or truncated:
                total_rewards.append(episode_reward)
                total_steps.append(step)
                
                if terminated:
                    success_count += 1
                    print(f"  ✓ 成功！步数: {step}, 奖励: {episode_reward:.2f}")
                else:
                    print(f"  ✗ 超时。步数: {step}, 奖励: {episode_reward:.2f}")
                break
    
    env.close()
    
    # 显示统计
    print("\n" + "=" * 70)
    print("演示统计")
    print("=" * 70)
    print(f"总回合数: {n_episodes}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_count / n_episodes * 100:.1f}%")
    print(f"平均奖励: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"平均步数: {sum(total_steps) / len(total_steps):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='演示训练好的模型')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（默认：最新模型）')
    parser.add_argument('--episodes', type=int, default=5,
                       help='演示回合数（默认：5）')
    parser.add_argument('--no-render', action='store_true',
                       help='不显示可视化')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，列出可用模型
    if args.model is None:
        print("\n可用的模型:")
        print("-" * 70)
        os.system("python list_models.py")
        print()
        
        use_latest = input("使用最新模型？(y/n，默认y): ").strip().lower()
        if use_latest == 'n':
            model_path = input("输入模型路径: ").strip()
            if not model_path:
                print("已取消")
                exit(0)
        else:
            model_path = None
    else:
        model_path = args.model
    
    play_model(
        model_path=model_path,
        n_episodes=args.episodes,
        render=not args.no_render
    )
