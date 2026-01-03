"""
多任务统一演示脚本

使用方式：
    python play.py --task arm --model runs/arm_20231219_120000/model
    python play.py --task walker --model runs/walker_20231219_120000/model
    python play.py --task cartpole --model runs/cartpole_20231219_120000/model
    python play.py --task double_pendulum --model runs/double_pendulum_20231219_120000/model
    python play.py --task walker --model runs/walker_20231219_120000/model --record  # 录制视频
"""
import argparse
import time
import os
from datetime import datetime
from envs import make_env
from stable_baselines3 import PPO
from configs import TASK_CONFIGS
import gymnasium as gym


def play(task, model_path, episodes=5, record=False, output_dir='videos', 
         enable_disturbance=False, disturbance_force=2.0, disturbance_prob=0.02, disturbance_type='cart_only'):
    """演示训练好的模型"""
    print("=" * 70)
    print(f"演示任务: {task}")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    print(f"演示回合: {episodes}")
    if record:
        print(f"录制视频: 是 (保存到 {output_dir}/)")
    if enable_disturbance:
        print(f"扰动测试: 启用 ({disturbance_type}, ±{disturbance_force}N, {disturbance_prob*100:.1f}%概率)")
    else:
        print(f"扰动测试: 关闭")
    print("=" * 70)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境
    config = TASK_CONFIGS[task]['env_config']
    env = make_env(task, render_mode='rgb_array' if record else 'human', config=config)
    
    # 设置扰动参数（如果指定）
    if hasattr(env, 'set_disturbance'):
        env.set_disturbance(enable_disturbance, disturbance_force, disturbance_prob, disturbance_type)
    
    # 如果需要录制，包装环境
    if record:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder = os.path.join(output_dir, f"{task}_{timestamp}")
        
        # 使用Gymnasium的RecordVideo包装器
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,  # 录制所有episode
            name_prefix=f"{task}_demo"
        )
        print(f"视频将保存到: {video_folder}/")
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n回合 {episode + 1}/{episodes}")
        
        while not done:
            # 预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # 渲染（如果不录制视频）
            if not record:
                env.render()
                # 控制帧率
                time.sleep(1.0 / 60.0)
        
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  步数: {steps}")
        if 'distance' in info:
            print(f"  距离: {info['distance']:.2f}m")
    
    env.close()
    
    if record:
        print(f"\n✓ 视频已保存到: {video_folder}/")
        print(f"  共录制 {episodes} 个回合")
    
    print("\n演示结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多任务RL演示')
    parser.add_argument('--task', type=str, required=True,
                       choices=list(TASK_CONFIGS.keys()),
                       help='任务名称')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--episodes', type=int, default=5,
                       help='演示回合数')
    parser.add_argument('--record', action='store_true',
                       help='录制视频')
    parser.add_argument('--output-dir', type=str, default='videos',
                       help='视频输出目录')
    
    # 扰动测试参数
    parser.add_argument('--disturbance', action='store_true',
                       help='启用扰动测试（测试模型鲁棒性）')
    parser.add_argument('--disturbance-force', type=float, default=2.0,
                       help='扰动力大小（N）')
    parser.add_argument('--disturbance-prob', type=float, default=0.02,
                       help='扰动概率（0-1）')
    parser.add_argument('--disturbance-type', type=str, default='cart_only',
                       choices=['cart_only', 'pole_only', 'both'],
                       help='扰动类型: cart_only(仅小车), pole_only(仅摆杆), both(同时)')
    
    args = parser.parse_args()
    
    play(args.task, args.model, args.episodes, args.record, args.output_dir,
         args.disturbance, args.disturbance_force, args.disturbance_prob, args.disturbance_type)
