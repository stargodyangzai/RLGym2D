"""
训练机械臂模型
"""
import os
import warnings
import json
from datetime import datetime
from collections import defaultdict

# 抑制pygame的deprecation警告
warnings.filterwarnings('ignore', category=UserWarning, module='pygame.pkgdata')

from simple_arm_env import SimpleArmEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
from config import get_config, print_config


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
    
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=10, verbose=1, log_reward_components=True, best_model_save_path=None, checkpoint_dir=None, checkpoint_freq=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.best_mean_reward = -np.inf
        self.log_reward_components = log_reward_components
        self.best_model_save_path = best_model_save_path  # 最佳模型保存路径
        self.checkpoint_dir = checkpoint_dir  # checkpoint保存目录
        self.checkpoint_freq = checkpoint_freq  # 每N次评估保存一次
        self.eval_count = 0  # 评估计数器
        self.checkpoint_count = 0  # checkpoint计数器
    
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
                        wandb.run.summary['best_mean_reward'] = mean_reward
                        wandb.run.summary['best_success_rate'] = success_rate
                        wandb.run.summary['best_timestep'] = self.num_timesteps
                except:
                    pass
            
            # 每次评估都保存checkpoint
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


def make_env(rank, seed=0, env_config=None, reward_config=None):
    """创建环境的工厂函数"""
    def _init():
        env = SimpleArmEnv(render_mode=None, config=env_config, reward_config=reward_config)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(config=None, preset=None, n_envs=None):
    """训练模型"""
    if config is None:
        config = get_config(preset)
    
    train_cfg = config['training']['parallel']
    ppo_cfg = config['ppo']
    
    if n_envs is None:
        n_envs = train_cfg['n_envs']
    
    # 计算total_timesteps（支持两种配置方式）
    if 'n_iterations' in train_cfg:
        # 方式1：直接指定迭代次数
        n_iterations = train_cfg['n_iterations']
        total_timesteps = n_iterations * ppo_cfg['n_steps'] * n_envs
        print(f"\n配置: {n_iterations}次迭代 × {ppo_cfg['n_steps']}步 × {n_envs}环境 = {total_timesteps:,}步")
    else:
        # 方式2：指定总步数（向后兼容）
        total_timesteps = train_cfg.get('total_timesteps', 1000000)
        n_iterations = total_timesteps // (ppo_cfg['n_steps'] * n_envs)
        print(f"\n配置: {total_timesteps:,}步 ≈ {n_iterations}次迭代")
    
    train_cfg['total_timesteps'] = total_timesteps  # 更新配置
    
    # 验证环境数是否合理
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    max_recommended = cpu_count * 4
    
    if n_envs > max_recommended:
        print(f"\n{'='*70}")
        print(f"⚠️  警告：并行环境数过多！")
        print(f"{'='*70}")
        print(f"  请求的环境数: {n_envs}")
        print(f"  CPU核心数: {cpu_count}")
        print(f"  推荐环境数: {cpu_count * 2}")
        print(f"  最大推荐: {max_recommended}")
        print(f"\n  过多的环境会导致:")
        print(f"    - 内存耗尽")
        print(f"    - 系统卡死")
        print(f"    - 训练崩溃")
        print(f"{'='*70}")
        
        response = input(f"\n是否继续？(y/n，推荐输入 n): ").strip().lower()
        if response != 'y':
            print(f"\n建议使用 {cpu_count * 2} 个环境")
            return None, None
        else:
            print("\n⚠️  继续使用过多环境，风险自负...")
    
    # 创建带时间戳的运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = "2d_arm_rl"
    run_name = f"{project_name}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"训练 ({n_envs} 个并行环境)")
    print("=" * 70)
    print(f"运行名称: {run_name}")
    print(f"保存目录: {run_dir}/")
    print("=" * 70)
    
    # 检测并显示设备信息
    import torch
    device = config['device']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "=" * 60)
    print("训练设备信息")
    print("=" * 60)
    if device == 'cuda' and torch.cuda.is_available():
        print(f"✓ 使用 GPU 训练")
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  PyTorch设备: {device}")
        print(f"  并行环境数: {n_envs}")
        print(f"  预计加速: 5-10倍（相比CPU）")
    else:
        print(f"✓ 使用 CPU 训练")
        print(f"  并行环境数: {n_envs}")
        print(f"  提示: 安装CUDA版PyTorch可获得5-10倍加速")
    print("=" * 60 + "\n")
    
    # 创建并行训练环境
    env = SubprocVecEnv([make_env(i, env_config=config['env'], reward_config=config['reward']) for i in range(n_envs)])
    
    # 创建评估环境
    eval_env = SimpleArmEnv(render_mode=None, config=config['env'], reward_config=config['reward'])
    
    # 保存配置到运行目录
    config_save = {
        'run_name': run_name,
        'timestamp': timestamp,
        'n_envs': n_envs,
        'env_config': config['env'],
        'reward_config': config['reward'],
        'ppo_config': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                       for k, v in ppo_cfg.items()},
        'training_config': train_cfg,
        'network_config': {
            'net_arch': config['network']['policy_kwargs']['net_arch']
        }
    }
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    
    # 初始化 W&B（如果启用）
    wandb_run = None
    if config.get('wandb', {}).get('enabled', False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                name=config['wandb']['name'] or run_name,
                tags=config['wandb']['tags'],
                notes=config['wandb']['notes'],
                config=config_save,
                sync_tensorboard=config['wandb']['sync_tensorboard'],
            )
            print("✓ W&B 日志已启用")
            print(f"  项目: {config['wandb']['project']}")
            print(f"  运行: {wandb_run.name}")
            print(f"  链接: {wandb_run.url}\n")
        except ImportError:
            print("⚠ W&B 未安装，跳过")
            print("  安装: pip install wandb\n")
        except Exception as e:
            print(f"⚠ W&B 初始化失败: {e}\n")
    
    # 创建模型（TensorBoard日志保存到运行目录）
    tensorboard_log = os.path.join(run_dir, "tensorboard")
    model = PPO(
        ppo_cfg['policy'],
        env,
        verbose=ppo_cfg['verbose'],
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        policy_kwargs=config['network']['policy_kwargs'],
        tensorboard_log=tensorboard_log,
        device=device
    )
    
    # 创建最佳模型保存目录
    best_model_dir = os.path.join(run_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 创建checkpoint保存目录
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建回调列表
    reward_logger = RewardComponentLogger(log_freq=100, verbose=0)
    performance_callback = PerformanceCallback(
        eval_env=eval_env,
        eval_freq=train_cfg['eval_freq'],
        n_eval_episodes=train_cfg['n_eval_episodes'],
        verbose=1,
        best_model_save_path=best_model_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=train_cfg.get('checkpoint_freq', 1)
    )
    
    # 组合回调
    callbacks = CallbackList([reward_logger, performance_callback])
    
    # 训练
    print(f"\n开始训练...")
    print(f"PPO迭代次数: {n_iterations}")
    print(f"总环境步数: {train_cfg['total_timesteps']:,}")
    print(f"每次迭代: {ppo_cfg['n_steps']} × {n_envs} = {ppo_cfg['n_steps'] * n_envs:,}步")
    print(f"评估频率: 每{train_cfg['eval_freq']}步")
    print(f"奖励组件记录: 每100个episode\n")
    model.learn(total_timesteps=train_cfg['total_timesteps'], progress_bar=True, callback=callbacks)
    
    # 保存模型到运行目录
    model_path = os.path.join(run_dir, "model")
    model.save(model_path)
    
    # 保存训练摘要
    best_model_path = os.path.join(best_model_dir, "best_model")
    summary = {
        'run_name': run_name,
        'timestamp': timestamp,
        'total_timesteps': train_cfg['total_timesteps'],
        'n_envs': n_envs,
        'device': device,
        'final_model_path': f"{model_path}.zip",
        'best_model_path': f"{best_model_path}.zip",
        'tensorboard_log': tensorboard_log,
        'best_reward': performance_callback.best_mean_reward if hasattr(performance_callback, 'best_mean_reward') else None,
    }
    
    with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 创建 README
    readme_content = f"""# 训练运行: {run_name}

## 基本信息
- **运行时间**: {timestamp}
- **训练步数**: {train_cfg['total_timesteps']:,}
- **并行环境**: {n_envs}
- **设备**: {device}

## 文件说明
- `model.zip` - 最终模型（训练结束时）
- `best_model/best_model.zip` - 最佳模型（评估奖励最高）⭐
- `checkpoints/` - 每次评估都保存checkpoint
- `config.json` - 完整配置
- `summary.json` - 训练摘要
- `tensorboard/` - TensorBoard 日志

## 如何使用

### 演示最佳模型（推荐）
```bash
python play.py --model {best_model_path}
```

### 演示最终模型
```bash
python play.py --model {model_path}
```

### 演示某个checkpoint
```bash
python play.py --model {os.path.join(run_dir, 'checkpoints', 'ckpt_iter_XXX_reward_YYY_success_ZZZ')}
```

### 评估模型
```bash
python evaluate_performance.py
# 输入路径: {model_path}
```

### 查看训练日志
```bash
tensorboard --logdir={tensorboard_log}
```
"""
    
    with open(os.path.join(run_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"运行目录: {run_dir}/")
    print(f"  ├── model.zip                    (最终模型)")
    print(f"  ├── best_model/best_model.zip    (最佳模型) ⭐")
    print(f"  ├── checkpoints/                 (定期checkpoint)")
    print(f"  ├── config.json                  (配置)")
    print(f"  ├── summary.json                 (摘要)")
    print(f"  ├── README.md                    (说明)")
    print(f"  └── tensorboard/                 (日志)")
    print("=" * 70)
    print(f"\n最佳评估奖励: {performance_callback.best_mean_reward:.2f}")
    print(f"推荐使用最佳模型: {best_model_path}.zip")
    print(f"保存的checkpoint数量: {performance_callback.checkpoint_count}")
    print("=" * 70)
    
    # 关闭 W&B
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
            print("✓ W&B 日志已上传")
        except:
            pass
    
    env.close()
    eval_env.close()
    return model, run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练2D机械臂模型')
    parser.add_argument('--envs', type=int, default=None,
                       help='并行环境数量（覆盖config.py中的设置）')
    parser.add_argument('--play', action='store_true',
                       help='训练后立即演示')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("2D机械臂强化学习训练")
    print("=" * 70)
    
    # 直接使用config.py中的配置
    config = get_config()
    print("\n✓ 使用 config.py 中的配置")
    print_config(config)
    
    # 确定环境数
    if args.envs is None:
        args.envs = config['training']['parallel']['n_envs']
    else:
        print(f"\n⚠️  命令行参数覆盖: 使用 {args.envs} 个并行环境")
    
    # 开始训练
    result = train(config=config, n_envs=args.envs)
    
    if result is None or result[0] is None:
        print("\n训练已取消或失败")
    else:
        model, run_dir = result
        
        # 询问是否演示
        if args.play:
            play = 'y'
        else:
            play = input("\n是否演示训练结果？(y/n): ").strip().lower()
        
        if play == 'y':
            # 优先使用最佳模型
            best_model_path = os.path.join(run_dir, "best_model", "best_model")
            if os.path.exists(f"{best_model_path}.zip"):
                demo_model = best_model_path
                print(f"\n启动演示（使用最佳模型）...")
            else:
                demo_model = os.path.join(run_dir, "model")
                print(f"\n启动演示（使用最终模型）...")
            os.system(f"python play.py --model {demo_model} --episodes 3")
