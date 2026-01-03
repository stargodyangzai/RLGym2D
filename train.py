"""
多任务统一训练脚本

支持的任务：
- arm: 2D机械臂到达目标
- walker: 2D火柴人行走
- cartpole: 倒立摆平衡
- double_pendulum: 二阶倒立摆平衡

使用方式：
    # 新训练
    python train.py --task arm --envs 32
    python train.py --task walker --envs 16
    python train.py --task cartpole --envs 8
    python train.py --task double_pendulum --envs 16
    
    # 继续训练
    python train.py --task double_pendulum --continue-from runs/xxx/best_model/best_model.zip
    python train.py --task double_pendulum --continue-from runs/xxx/best_model/best_model.zip --continue-iterations 200
    
    # 训练后立即演示
    python train.py --task double_pendulum --envs 16 --play
"""
import os
import sys
import argparse
import warnings
import json
from datetime import datetime
from collections import defaultdict

# 设置环境变量抑制警告（必须在导入其他模块前）
os.environ['PYTHONWARNINGS'] = 'ignore'

from envs import make_env
from configs import TASK_CONFIGS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
import torch

# 抑制所有警告
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


class RewardComponentLogger(BaseCallback):
    """记录训练过程中的奖励组件"""
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = defaultdict(list)
        self.episode_count = 0
        self.current_episode_rewards = defaultdict(float)
    
    def _on_step(self):
        # 获取当前环境的info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # 累积当前episode的奖励组件
            if 'reward_components' in info:
                for comp_name, comp_value in info['reward_components'].items():
                    self.current_episode_rewards[comp_name] += float(comp_value)
            
            # 检查episode是否结束
            dones = self.locals.get('dones', [False])
            if dones[0]:
                self.episode_count += 1
                
                # 记录这个episode的奖励组件
                for comp_name, comp_value in self.current_episode_rewards.items():
                    self.episode_rewards[comp_name].append(comp_value)
                
                # 每log_freq个episode记录一次平均值
                if self.episode_count % self.log_freq == 0:
                    # 计算平均值
                    avg_rewards = {}
                    total_reward = 0
                    for comp_name, values in self.episode_rewards.items():
                        if len(values) > 0:
                            avg_value = np.mean(values[-self.log_freq:])
                            avg_rewards[comp_name] = avg_value
                            total_reward += avg_value
                    
                    # 计算贡献百分比
                    contributions = {}
                    if total_reward != 0:
                        for comp_name, avg_value in avg_rewards.items():
                            contributions[comp_name] = (avg_value / total_reward) * 100
                    
                    # 记录到TensorBoard
                    for comp_name, avg_value in avg_rewards.items():
                        self.logger.record(f'train/reward_component/{comp_name}', avg_value)
                        if comp_name in contributions:
                            self.logger.record(f'train/reward_contribution/{comp_name}_percent', contributions[comp_name])
                    
                    # 记录到W&B
                    try:
                        import wandb
                        if wandb.run is not None:
                            log_dict = {}
                            for comp_name, avg_value in avg_rewards.items():
                                log_dict[f'train/reward_component/{comp_name}'] = avg_value
                                if comp_name in contributions:
                                    log_dict[f'train/reward_contribution/{comp_name}_percent'] = contributions[comp_name]
                            wandb.log(log_dict, step=self.num_timesteps)
                    except:
                        pass
                
                # 重置当前episode的累积
                self.current_episode_rewards = defaultdict(float)
        
        return True


class PerformanceCallback(BaseCallback):
    """自定义回调：记录训练性能指标"""
    
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, verbose=1, 
                 log_reward_components=True, best_model_save_path=None, 
                 checkpoint_dir=None, checkpoint_freq=10):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.best_mean_reward = -np.inf
        self.log_reward_components = log_reward_components
        self.best_model_save_path = best_model_save_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.eval_count = 0
        self.checkpoint_count = 0
    
    def _on_step(self):
        # 使用 num_timesteps 而不是 n_calls，确保按环境步数评估
        if self.num_timesteps % self.eval_freq == 0:
            # 评估当前策略
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            reward_components_sum = {}
            
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                episode_components = {}
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                    
                    # 累积奖励组件
                    if self.log_reward_components and 'reward_components' in info:
                        for comp_name, comp_value in info['reward_components'].items():
                            if comp_name not in episode_components:
                                episode_components[comp_name] = 0
                            episode_components[comp_name] += float(comp_value)
                    
                    if terminated:
                        # 检查是否有明确的成功标志
                        if 'success' in info and info['success']:
                            success_count += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 累积所有回合的组件
                for comp_name, comp_value in episode_components.items():
                    if comp_name not in reward_components_sum:
                        reward_components_sum[comp_name] = []
                    reward_components_sum[comp_name].append(comp_value)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = success_count / self.n_eval_episodes * 100
            
            self.evaluations_results.append(mean_reward)
            self.evaluations_timesteps.append(self.num_timesteps)
            
            # 记录到W&B
            try:
                import wandb
                if wandb.run is not None:
                    log_dict = {
                        'eval/mean_reward': mean_reward,
                        'eval/std_reward': std_reward,
                        'eval/mean_length': mean_length,
                        'eval/success_rate': success_rate,
                        'eval/best_mean_reward': self.best_mean_reward,
                    }
                    
                    # 添加奖励组件
                    if self.log_reward_components and reward_components_sum:
                        for comp_name, comp_values in reward_components_sum.items():
                            log_dict[f'eval/reward_component/{comp_name}'] = np.mean(comp_values)
                    
                    wandb.log(log_dict, step=self.num_timesteps)
            except:
                pass
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"评估 @ {self.num_timesteps} 步:")
                print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"  平均长度: {mean_length:.1f}")
                # 只在有成功事件时显示成功率
                if success_count > 0:
                    print(f"  成功率: {success_rate:.1f}%")
                print(f"{'='*60}\n")
            
            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"🌟 新的最佳模型！奖励: {mean_reward:.2f}")
                
                if self.best_model_save_path is not None:
                    best_model_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(best_model_path)
                    if self.verbose > 0:
                        print(f"   已保存到: {best_model_path}.zip")
                
                # 记录最佳模型到W&B
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.run.summary['best_mean_reward'] = float(mean_reward)  # 确保是Python float
                        wandb.run.summary['best_success_rate'] = float(success_rate)  # 确保是Python float
                        wandb.run.summary['best_timestep'] = int(self.num_timesteps)  # 确保是Python int
                except:
                    pass
            
            # 定期保存checkpoint
            self.eval_count += 1
            if self.checkpoint_dir is not None and self.eval_count % self.checkpoint_freq == 0:
                # 计算当前迭代次数（近似）
                iteration = self.num_timesteps // (self.model.n_steps * self.model.n_envs)
                
                # 创建checkpoint文件名：包含迭代次数、奖励、成功率
                self.checkpoint_count += 1
                checkpoint_name = f"ckpt_iter_{iteration:03d}_reward_{mean_reward:+.1f}_success_{success_rate:.2f}.zip"
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
                
                # 保存checkpoint
                self.model.save(checkpoint_path)
                
                if self.verbose > 0:
                    print(f"💾 保存checkpoint #{self.checkpoint_count}: {checkpoint_name}")
        
        return True


def make_env_fn(task, rank, seed=0):
    """创建环境的工厂函数"""
    def _init():
        # 在每个子进程中抑制警告
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='pygame.pkgdata')
        
        # 合并env_config和reward_config
        task_config = TASK_CONFIGS[task]
        config = task_config['env_config'].copy()
        if 'reward_config' in task_config:
            config['reward_config'] = task_config['reward_config']
        
        env = make_env(task, render_mode=None, config=config)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(task, n_envs=None, device=None, continue_from=None, continue_iterations=None):
    """训练模型"""
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    
    config = TASK_CONFIGS[task]
    train_cfg = config['training_config']
    ppo_cfg = config['ppo_config']
    
    if n_envs is None:
        n_envs = train_cfg['n_envs']
    
    # 计算总步数
    if continue_iterations is not None:
        # 使用指定的继续训练迭代次数
        n_iterations = continue_iterations
        total_timesteps = n_iterations * ppo_cfg['n_steps'] * n_envs
    else:
        # 使用配置文件中的默认迭代次数
        n_iterations = train_cfg['n_iterations']
        total_timesteps = n_iterations * ppo_cfg['n_steps'] * n_envs
    
    print("=" * 70)
    print(f"训练任务: {task}")
    print("=" * 70)
    print(f"迭代次数: {n_iterations}")
    print(f"并行环境: {n_envs}")
    print(f"总步数: {total_timesteps:,}")
    print("=" * 70)
    
    # 设备选择
    if device is None:
        device = 'cpu'
    elif device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n使用设备: {device}")
    
    if device == 'cpu':
        print("提示: MLP策略在CPU上训练通常更高效")
        if torch.cuda.is_available():
            print(f"检测到GPU: {torch.cuda.get_device_name(0)}（未使用）")
            print("如需使用GPU，请添加参数: --device cuda")
    elif device == 'cuda':
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print("⚠️  注意: MLP策略在GPU上可能比CPU慢")
        else:
            print("⚠️  警告: 未检测到GPU，将使用CPU")
            device = 'cpu'
    
    print()
    
    # 创建运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{task}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"运行目录: {run_dir}/\n")
    
    # 创建并行环境
    env = SubprocVecEnv([make_env_fn(task, i) for i in range(n_envs)])
    
    # 创建评估环境
    task_config = TASK_CONFIGS[task]
    eval_config = task_config['env_config'].copy()
    if 'reward_config' in task_config:
        eval_config['reward_config'] = task_config['reward_config']
    eval_env = make_env(task, render_mode=None, config=eval_config)
    
    # 保存配置
    config_save = {
        'run_name': run_name,
        'timestamp': timestamp,
        'task': task,
        'n_envs': n_envs,
        'device': device,
        'env_config': config['env_config'],
        'ppo_config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                       for k, v in ppo_cfg.items()},
        'training_config': train_cfg,
        'network_config': config['network_config'],
    }
    
    if 'reward_config' in config:
        config_save['reward_config'] = config['reward_config']
    
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    
    # 创建或加载模型
    tensorboard_log = os.path.join(run_dir, "tensorboard")
    
    if continue_from is not None:
        # 继续训练：加载已有模型
        if not os.path.exists(continue_from):
            raise FileNotFoundError(f"模型文件不存在: {continue_from}")
        
        print(f"🔄 从已有模型继续训练: {continue_from}")
        model = PPO.load(continue_from, env=env, device=device)
        
        # 更新tensorboard日志路径
        model.tensorboard_log = tensorboard_log
        
        print(f"✅ 模型加载成功，将继续训练 {n_iterations} 次迭代 ({total_timesteps:,} 步)")
        
    else:
        # 新训练：创建新模型
        print(f"🆕 创建新模型，开始训练 {n_iterations} 次迭代 ({total_timesteps:,} 步)")
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=ppo_cfg['learning_rate'],
            n_steps=ppo_cfg['n_steps'],
            batch_size=ppo_cfg['batch_size'],
            n_epochs=ppo_cfg['n_epochs'],
            gamma=ppo_cfg['gamma'],
            ent_coef=ppo_cfg['ent_coef'],
            policy_kwargs={'net_arch': config['network_config']['net_arch']},
            tensorboard_log=tensorboard_log,
            device=device
        )
    
    # 创建保存目录
    best_model_dir = os.path.join(run_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建回调
    reward_logger = RewardComponentLogger(log_freq=100, verbose=0)
    performance_callback = PerformanceCallback(
        eval_env=eval_env,
        eval_freq=ppo_cfg['n_steps'] * n_envs,
        n_eval_episodes=train_cfg.get('n_eval_episodes', 10),
        verbose=1,
        best_model_save_path=best_model_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=train_cfg.get('checkpoint_freq', 10)
    )
    callbacks = CallbackList([reward_logger, performance_callback])
    
    # 训练
    print("开始训练...")
    print(f"PPO迭代次数: {n_iterations}")
    print(f"总环境步数: {total_timesteps:,}")
    print(f"Checkpoint频率: 每{train_cfg.get('checkpoint_freq', 10)}次评估\n")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    
    # 保存模型
    model_path = os.path.join(run_dir, "model")
    model.save(model_path)
    
    # 保存摘要
    best_model_path = os.path.join(best_model_dir, "best_model")
    summary = {
        'run_name': run_name,
        'task': task,
        'total_timesteps': total_timesteps,
        'best_reward': float(performance_callback.best_mean_reward),  # 转换为Python float
        'checkpoint_count': performance_callback.checkpoint_count,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 创建README
    readme = f"""# 训练运行: {run_name}

## 基本信息
- 任务: {task}
- 训练步数: {total_timesteps:,}
- 设备: {device}

## 使用
```bash
python play.py --task {task} --model {best_model_path}
```
"""
    with open(os.path.join(run_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"运行目录: {run_dir}/")
    print(f"最佳奖励: {performance_callback.best_mean_reward:.2f}")
    print(f"Checkpoint数量: {performance_callback.checkpoint_count}")
    print("=" * 70)
    
    env.close()
    eval_env.close()
    return model, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多任务RL训练')
    parser.add_argument('--task', type=str, required=True,
                       choices=list(TASK_CONFIGS.keys()),
                       help='任务名称')
    parser.add_argument('--envs', type=int, default=None,
                       help='并行环境数量')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'auto'],
                       help='训练设备')
    parser.add_argument('--play', action='store_true',
                       help='训练后立即演示')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='从指定模型继续训练 (模型路径，如: runs/xxx/best_model/best_model.zip)')
    parser.add_argument('--continue-iterations', type=int, default=None,
                       help='继续训练的迭代次数 (默认使用配置文件中的设置)')
    
    args = parser.parse_args()
    
    # 验证继续训练参数
    if args.continue_from is not None:
        if not os.path.exists(args.continue_from):
            print(f"❌ 错误: 模型文件不存在: {args.continue_from}")
            sys.exit(1)
        print(f"🔄 继续训练模式")
        print(f"   模型路径: {args.continue_from}")
        if args.continue_iterations:
            print(f"   训练迭代: {args.continue_iterations} 次")
        else:
            print(f"   训练迭代: 使用配置文件默认值")
    
    model, run_dir = train(args.task, args.envs, args.device, args.continue_from, args.continue_iterations)
    
    if args.play:
        print("\n启动演示...")
        best_model = os.path.join(run_dir, "best_model", "best_model")
        if os.path.exists(f"{best_model}.zip"):
            os.system(f"python play.py --task {args.task} --model {best_model}")
