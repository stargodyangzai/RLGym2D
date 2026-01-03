"""
二阶倒立摆环境（Double Pendulum）

任务：控制小车左右移动，保持两个连接的摆杆都直立
- 观察：小车位置、速度、两个摆杆的角度、角速度
- 动作：向左或向右施加力
- 目标：保持两个摆杆都直立，小车在轨道中央

相比单摆，二阶倒立摆具有：
1. 更高的控制难度
2. 更复杂的动力学
3. 更丰富的混沌行为
4. 更强的非线性特性
"""
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np


class DoublePendulumEnv(gym.Env):
    """
    二阶倒立摆环境
    
    结构：
    - 小车（cart）：可以左右移动
    - 第一摆杆（pole1）：通过关节连接到小车
    - 第二摆杆（pole2）：通过关节连接到第一摆杆的末端
    - 轨道（track）：限制小车移动范围
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, config=None):
        """
        初始化二阶倒立摆环境
        
        Args:
            render_mode: 渲染模式
            config: 配置字典
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.config = config or {}
        
        # 物理参数
        self.dt = self.config.get('dt', 1.0/60.0)
        self.max_steps = self.config.get('max_steps', 1000)
        self.gravity = self.config.get('gravity', 9.81)
        
        # 二阶倒立摆参数（可配置）
        self.cart_mass = self.config.get('cart_mass', 1.0)  # kg
        self.pole1_mass = self.config.get('pole1_mass', 0.15)  # kg (第一摆杆)
        self.pole2_mass = self.config.get('pole2_mass', 0.08)  # kg (第二摆杆)
        self.pole1_length = self.config.get('pole1_length', 0.9)  # m (第一摆杆半长)
        self.pole2_length = self.config.get('pole2_length', 0.7)  # m (第二摆杆半长)
        self.force_mag = self.config.get('force_mag', 18.0)  # N
        
        # 轨道限制
        self.track_length = self.config.get('position_threshold', 3.0) * 2  # 更长的轨道
        
        # 扰动配置
        self.enable_disturbance = self.config.get('enable_disturbance', False)
        self.disturbance_force_range = self.config.get('disturbance_force_range', 3.0)
        self.disturbance_probability = self.config.get('disturbance_probability', 0.01)
        self.disturbance_type = self.config.get('disturbance_type', 'cart_only')
        
        # 成功条件（更严格）
        self.angle_threshold = self.config.get('angle_threshold', 15)  # 度
        self.position_threshold = self.config.get('position_threshold', 3.0)  # m
        
        # 物理引擎
        self.space = None
        self.cart = None
        self.pole1 = None
        self.pole2 = None
        self.joint1 = None  # 小车-摆杆1
        self.joint2 = None  # 摆杆1-摆杆2
        
        # 渲染
        self.screen = None
        self.clock = None
        self.draw_options = None
        self.screen_width = 1000  # 更宽的屏幕
        self.screen_height = 700   # 更高的屏幕
        
        # 状态
        self.current_step = 0
        self.last_disturbance = 0.0
        self.disturbance_counter = 0
        
        # 观察空间：[cart_pos, cart_vel, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel]
        high = np.array([
            self.position_threshold * 2,  # cart_pos
            np.finfo(np.float32).max,     # cart_vel
            np.pi,                        # pole1_angle
            np.finfo(np.float32).max,     # pole1_angular_vel
            np.pi,                        # pole2_angle
            np.finfo(np.float32).max,     # pole2_angular_vel
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # 动作空间：连续力 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 初始化物理引擎
        self._init_physics()
    
    def _init_physics(self):
        """初始化物理引擎"""
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity * 100)  # Pymunk uses pixels
    
    def _create_double_pendulum(self):
        """创建二阶倒立摆"""
        scale = 100  # 100像素 = 1米
        
        # 屏幕中心
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # 1. 创建小车
        cart_width = 0.6 * scale
        cart_height = 0.3 * scale
        cart_moment = pymunk.moment_for_box(self.cart_mass, (cart_width, cart_height))
        self.cart = pymunk.Body(self.cart_mass, cart_moment)
        self.cart.position = (center_x, center_y)
        
        cart_shape = pymunk.Poly.create_box(self.cart, (cart_width, cart_height))
        cart_shape.friction = 0.5
        cart_shape.color = (100, 150, 200, 255)
        self.space.add(self.cart, cart_shape)
        
        # 限制小车只能水平移动
        groove_start = (center_x - self.track_length/2 * scale, center_y)
        groove_end = (center_x + self.track_length/2 * scale, center_y)
        groove = pymunk.GrooveJoint(self.space.static_body, self.cart, groove_start, groove_end, (0, 0))
        self.space.add(groove)
        
        # 2. 创建第一摆杆
        pole1_length_px = self.pole1_length * 2 * scale  # 总长度
        pole1_moment = pymunk.moment_for_segment(
            self.pole1_mass, (0, 0), (0, -pole1_length_px), 6
        )
        self.pole1 = pymunk.Body(self.pole1_mass, pole1_moment)
        self.pole1.position = (center_x, center_y - pole1_length_px/2)
        
        pole1_shape = pymunk.Segment(self.pole1, (0, pole1_length_px/2), (0, -pole1_length_px/2), 6)
        pole1_shape.friction = 0.5
        pole1_shape.color = (200, 100, 100, 255)
        self.space.add(self.pole1, pole1_shape)
        
        # 3. 创建第二摆杆
        pole2_length_px = self.pole2_length * 2 * scale  # 总长度
        pole2_moment = pymunk.moment_for_segment(
            self.pole2_mass, (0, 0), (0, -pole2_length_px), 4
        )
        self.pole2 = pymunk.Body(self.pole2_mass, pole2_moment)
        # 第二摆杆连接到第一摆杆的末端
        pole1_end_y = center_y - pole1_length_px
        self.pole2.position = (center_x, pole1_end_y - pole2_length_px/2)
        
        pole2_shape = pymunk.Segment(self.pole2, (0, pole2_length_px/2), (0, -pole2_length_px/2), 4)
        pole2_shape.friction = 0.5
        pole2_shape.color = (100, 200, 100, 255)
        self.space.add(self.pole2, pole2_shape)
        
        # 4. 创建关节
        # 小车-第一摆杆的旋转关节
        self.joint1 = pymunk.PivotJoint(self.cart, self.pole1, (0, 0), (0, pole1_length_px/2))
        self.joint1.collide_bodies = False
        self.space.add(self.joint1)
        
        # 第一摆杆-第二摆杆的旋转关节
        self.joint2 = pymunk.PivotJoint(self.pole1, self.pole2, (0, -pole1_length_px/2), (0, pole2_length_px/2))
        self.joint2.collide_bodies = False
        self.space.add(self.joint2)
    
    def _get_obs(self):
        """获取观察"""
        scale = 100
        center_x = self.screen_width // 2
        
        # 小车位置和速度
        cart_pos = (self.cart.position.x - center_x) / scale
        cart_vel = self.cart.velocity.x / scale
        
        # 第一摆杆角度和角速度（0度是向上）
        pole1_angle = self.pole1.angle
        pole1_angular_vel = self.pole1.angular_velocity
        
        # 第二摆杆角度和角速度
        pole2_angle = self.pole2.angle
        pole2_angular_vel = self.pole2.angular_velocity
        
        obs = np.array([
            cart_pos, cart_vel, 
            pole1_angle, pole1_angular_vel,
            pole2_angle, pole2_angular_vel
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 应用控制力到小车
        control_force = np.clip(action[0], -1.0, 1.0) * self.force_mag
        
        # 随机扰动
        disturbance = 0.0
        if self.enable_disturbance and self.np_random is not None:
            if self.np_random.random() < self.disturbance_probability:
                disturbance = self.np_random.uniform(
                    -self.disturbance_force_range, 
                    self.disturbance_force_range
                )
                self.last_disturbance = disturbance
                self.disturbance_counter += 1
                print(f"🌪️ 步骤 {self.current_step}: 扰动 {disturbance:.2f}N (第{self.disturbance_counter}次)")
        
        # 应用力
        total_force = control_force + disturbance
        self.cart.apply_force_at_local_point((total_force * 100, 0), (0, 0))
        
        # 物理仿真
        self.space.step(self.dt)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward, info = self._compute_reward(obs)
        
        # 添加扰动信息到info
        info['disturbance'] = float(disturbance)
        info['control_force'] = float(control_force)
        info['total_force'] = float(total_force)
        
        # 检查终止
        terminated = self._check_termination(obs)
        truncated = self.current_step >= self.max_steps
        
        # 成功标志
        info['success'] = truncated and not terminated
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs):
        """
        计算奖励 - 使用平滑乘法奖励避免"弃车保帅"
        
        核心改进：
        1. 使用高斯函数替代阶梯函数 -> 消除梯度死区
        2. 引入角速度惩罚 -> 杀死"旋转刷分"
        3. 平滑的乘法关系 -> 鼓励协调控制
        
        Reward = Pole1_Total × Pole2_Total × Position_Factor
        其中 Pole_Total = Angle_Status × (base + velocity_weight × Velocity_Status)
        """
        cart_pos, cart_vel, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel = obs
        
        # 从配置中获取参数
        reward_config = self.config.get('reward_config', {})
        use_multiplicative = reward_config.get('use_multiplicative', True)
        use_smooth_gaussian = reward_config.get('use_smooth_gaussian', True)  # 是否使用平滑高斯
        
        pole1_angle_deg = abs(np.degrees(pole1_angle))
        pole2_angle_deg = abs(np.degrees(pole2_angle))
        
        if use_multiplicative:
            if use_smooth_gaussian:
                # ========== 平滑高斯乘法奖励（推荐）==========
                # 从配置读取高斯参数
                angle1_sigma = reward_config.get('angle1_sigma', 0.10)  # 第一摆杆角度容忍度
                angle2_sigma = reward_config.get('angle2_sigma', 0.15)  # 第二摆杆角度容忍度
                vel1_sigma = reward_config.get('vel1_sigma', 5.0)      # 第一摆杆角速度容忍度
                vel2_sigma = reward_config.get('vel2_sigma', 10.0)     # 第二摆杆角速度容忍度
                vel1_weight = reward_config.get('vel1_weight', 0.2)    # 第一摆杆速度权重
                vel2_weight = reward_config.get('vel2_weight', 0.3)    # 第二摆杆速度权重
                pos_sigma_factor = reward_config.get('pos_sigma_factor', 1.0)  # 位置容忍度因子
                
                # 1. 平滑的角度状态 (高斯分布) - 消除梯度死区
                # exp(-x²/σ²) 在任何地方都有导数，指引智能体向0度靠拢
                p1_angle_status = np.exp(-(pole1_angle**2) / angle1_sigma)  # 约18度时≈0.36
                p2_angle_status = np.exp(-(pole2_angle**2) / angle2_sigma)  # 对第二阶稍微宽容
                
                # 2. 核心改进：角速度抑制 - 杀死"旋转刷分"
                # 只有角度正且速度慢，才叫真的稳定
                v1_status = np.exp(-(pole1_angular_vel**2) / vel1_sigma)
                v2_status = np.exp(-(pole2_angular_vel**2) / vel2_sigma)  # 惩罚第二阶的旋转
                
                # 3. 组合状态 (乘法) - 将角度和速度状态结合
                # 速度占一部分权重，鼓励"静止的直立"而非"旋转经过直立"
                pole1_total = p1_angle_status * (1.0 - vel1_weight + vel1_weight * v1_status)
                pole2_total = p2_angle_status * (1.0 - vel2_weight + vel2_weight * v2_status)
                
                # 4. 位置因子 (平滑高斯)
                pos_status = np.exp(-(cart_pos**2) / (self.position_threshold**2 * pos_sigma_factor))
                
                # 5. 总奖励计算 - 核心乘法
                # 只有当两个摆杆都直立且不乱动时，分数才高
                base_reward = pole1_total * pole2_total * pos_status
                
                # 缩放到合理范围
                reward_scale = reward_config.get('reward_scale', 10.0)
                total_reward = base_reward * reward_scale
                
                # 6. 解决"不愿倾斜"的Trick：给第二阶彻底倒下时严厉惩罚
                # 但不给第一阶设硬门槛，鼓励它为了救P2而适度倾斜
                pole2_collapse_threshold = reward_config.get('pole2_collapse_threshold', 60)  # 度
                if pole2_angle_deg > pole2_collapse_threshold:
                    total_reward *= 0.1  # 第二阶彻底倒了，奖励直接打折
                
                info = {
                    'p1_angle_status': float(p1_angle_status),
                    'p2_angle_status': float(p2_angle_status),
                    'v1_status': float(v1_status),
                    'v2_status': float(v2_status),
                    'pole1_total': float(pole1_total),
                    'pole2_total': float(pole2_total),
                    'pos_status': float(pos_status),
                    'base_reward': float(base_reward),
                    'collapse_penalty': 0.1 if pole2_angle_deg > pole2_collapse_threshold else 1.0,
                    'cart_pos': float(cart_pos),
                    'pole1_angle_deg': float(pole1_angle_deg),
                    'pole2_angle_deg': float(pole2_angle_deg),
                    'pole1_angular_vel': float(pole1_angular_vel),
                    'pole2_angular_vel': float(pole2_angular_vel),
                    'reward_mode': 'multiplicative_smooth_gaussian'
                }
                
            else:
                # ========== 阶梯式乘法奖励（旧版，保留用于对比）==========
                # 1. 第一摆杆状态 [0, 1]，越接近垂直越接近1
                if pole1_angle_deg < 5:
                    pole1_status = 1.0
                elif pole1_angle_deg < 15:
                    pole1_status = 1.0 - (pole1_angle_deg - 5) / 10 * 0.3  # [1.0, 0.7]
                elif pole1_angle_deg < 30:
                    pole1_status = 0.7 - (pole1_angle_deg - 15) / 15 * 0.5  # [0.7, 0.2]
                else:
                    pole1_status = max(0.2 - (pole1_angle_deg - 30) / 30 * 0.2, 0.0)  # [0.2, 0.0]
                
                # 2. 第二摆杆状态 [0, 1]
                if pole2_angle_deg < 8:
                    pole2_status = 1.0
                elif pole2_angle_deg < 20:
                    pole2_status = 1.0 - (pole2_angle_deg - 8) / 12 * 0.4  # [1.0, 0.6]
                elif pole2_angle_deg < 40:
                    pole2_status = 0.6 - (pole2_angle_deg - 20) / 20 * 0.5  # [0.6, 0.1]
                else:
                    pole2_status = max(0.1 - (pole2_angle_deg - 40) / 30 * 0.1, 0.0)  # [0.1, 0.0]
                
                # 3. 位置因子 [0.5, 1.0]，越靠近中心越接近1
                position_factor = 1.0 - abs(cart_pos) / (self.position_threshold * 2) * 0.5
                position_factor = max(position_factor, 0.5)
                
                # 4. 速度因子 [0.8, 1.0]，速度越小越接近1
                velocity_factor = 1.0 - min(abs(cart_vel) / 5.0, 1.0) * 0.2
                
                # 核心乘法：任一摆杆倒下，总奖励趋近于0
                base_reward = pole1_status * pole2_status * position_factor * velocity_factor
                
                # 缩放到合理范围 [0, 10]
                total_reward = base_reward * 10.0
                
                # 额外奖励：两个摆杆都非常稳定时
                if pole1_angle_deg < 5 and pole2_angle_deg < 8:
                    total_reward += 2.0  # 稳定奖励
                
                info = {
                    'pole1_status': float(pole1_status),
                    'pole2_status': float(pole2_status),
                    'position_factor': float(position_factor),
                    'velocity_factor': float(velocity_factor),
                    'base_reward': float(base_reward),
                    'stability_bonus': 2.0 if (pole1_angle_deg < 5 and pole2_angle_deg < 8) else 0.0,
                    'cart_pos': float(cart_pos),
                    'pole1_angle_deg': float(pole1_angle_deg),
                    'pole2_angle_deg': float(pole2_angle_deg),
                    'pole1_angular_vel': float(pole1_angular_vel),
                    'pole2_angular_vel': float(pole2_angular_vel),
                    'reward_mode': 'multiplicative_stepwise'
                }
            
        else:
            # ========== 加法奖励模式（旧版，保留用于对比）==========
            pole1_weight = reward_config.get('pole1_weight', 1.5)
            pole2_weight = reward_config.get('pole2_weight', 3.0)
            coordination_weight = reward_config.get('coordination_weight', 1.0)
            
            # 第一摆杆角度奖励
            if pole1_angle_deg < 5:
                pole1_reward = pole1_weight * 2.0
            elif pole1_angle_deg < 15:
                pole1_reward = pole1_weight * (1.33 - (pole1_angle_deg - 5) * 0.067)
            else:
                pole1_reward = max(-pole1_weight * ((pole1_angle_deg - 15) / 10) ** 2, -pole1_weight * 5.33)
            
            # 第二摆杆角度奖励
            if pole2_angle_deg < 8:
                pole2_reward = pole2_weight * 0.67
            elif pole2_angle_deg < 20:
                pole2_reward = pole2_weight * (0.5 - (pole2_angle_deg - 8) * 0.027)
            else:
                pole2_reward = max(-pole2_weight * ((pole2_angle_deg - 20) / 15) ** 2, -pole2_weight * 1.67)
            
            # 协调奖励
            if pole1_angle_deg < 10 and pole2_angle_deg < 15:
                coordination_reward = coordination_weight
            else:
                coordination_reward = 0.0
            
            # 位置和速度惩罚
            position_penalty = -abs(cart_pos) * position_weight
            cart_vel_penalty = -abs(cart_vel) * cart_vel_weight
            
            total_reward = (pole1_reward + pole2_reward + coordination_reward + 
                           position_penalty + cart_vel_penalty)
            
            info = {
                'pole1_reward': float(pole1_reward),
                'pole2_reward': float(pole2_reward),
                'coordination_reward': float(coordination_reward),
                'position_penalty': float(position_penalty),
                'cart_vel_penalty': float(cart_vel_penalty),
                'cart_pos': float(cart_pos),
                'pole1_angle_deg': float(pole1_angle_deg),
                'pole2_angle_deg': float(pole2_angle_deg),
                'pole1_angular_vel': float(pole1_angular_vel),
                'pole2_angular_vel': float(pole2_angular_vel),
                'reward_mode': 'additive'
            }
        
        return total_reward, info
    
    def _check_termination(self, obs):
        """检查是否终止"""
        # 从配置中读取是否禁用终止条件
        disable_termination = self.config.get('disable_termination', False)
        
        if disable_termination:
            return False
        
        cart_pos, _, pole1_angle, _, pole2_angle, _ = obs
        
        # 小车超出轨道
        if abs(cart_pos) > self.position_threshold:
            return True
        
        # 任一摆杆倾斜太大
        pole1_angle_deg = abs(np.degrees(pole1_angle))
        pole2_angle_deg = abs(np.degrees(pole2_angle))
        
        if pole1_angle_deg > self.angle_threshold or pole2_angle_deg > self.angle_threshold:
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 清空物理空间
        for body in list(self.space.bodies):
            self.space.remove(body)
        for shape in list(self.space.shapes):
            self.space.remove(shape)
        for constraint in list(self.space.constraints):
            self.space.remove(constraint)
        
        # 重新创建二阶倒立摆
        self._create_double_pendulum()
        
        # 添加小的随机扰动
        if self.np_random is not None:
            self.pole1.angle = self.np_random.uniform(-0.03, 0.03)
            self.pole2.angle = self.np_random.uniform(-0.05, 0.05)
            self.cart.position = (
                self.cart.position.x + self.np_random.uniform(-20, 20),
                self.cart.position.y
            )
        
        # 重置步数和扰动记录
        self.current_step = 0
        self.last_disturbance = 0.0
        self.disturbance_counter = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def render(self):
        """渲染 - 美化版"""
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Double Pendulum - 二阶倒立摆")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 清屏
        self.screen.fill((255, 255, 255))
        
        # 绘制轨道
        scale = 100
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        track_start = center_x - self.track_length/2 * scale
        track_end = center_x + self.track_length/2 * scale
        
        # 轨道底座
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (track_start - 10, center_y + 20, track_end - track_start + 20, 10))
        # 轨道线
        pygame.draw.line(self.screen, (150, 150, 150), 
                        (track_start, center_y), (track_end, center_y), 3)
        
        # 绘制中心标记
        pygame.draw.line(self.screen, (200, 200, 200), 
                        (center_x, center_y - 15), (center_x, center_y + 15), 2)
        
        # 使用pymunk的debug绘制，但配置选项来隐藏质心
        if hasattr(self.draw_options, 'flags'):
            # 设置绘制选项，隐藏质心和其他不需要的元素
            self.draw_options.flags = (
                pymunk.pygame_util.DrawOptions.DRAW_SHAPES |
                pymunk.pygame_util.DrawOptions.DRAW_CONSTRAINTS
            )
        
        # 绘制物理对象
        self.space.debug_draw(self.draw_options)
        
        # 绘制连接点（关节）- 使用更明显的颜色和大小
        if self.joint1 and self.joint2:
            # 小车-摆杆1连接点（红色）
            joint1_pos = self.cart.position
            pygame.draw.circle(self.screen, (200, 50, 50), 
                             (int(joint1_pos.x), int(joint1_pos.y)), 6)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             (int(joint1_pos.x), int(joint1_pos.y)), 3)
            
            # 摆杆1-摆杆2连接点（蓝色）
            # 计算第二个关节的位置
            pole1_end = self.pole1.local_to_world((0, -self.pole1_length * 100))
            pygame.draw.circle(self.screen, (50, 50, 200), 
                             (int(pole1_end.x), int(pole1_end.y)), 5)
            pygame.draw.circle(self.screen, (255, 255, 255), 
                             (int(pole1_end.x), int(pole1_end.y)), 2)
        
        # UI信息面板
        if self.render_mode == 'human':
            # 半透明背景
            panel_surface = pygame.Surface((self.screen_width, 100), pygame.SRCALPHA)
            pygame.draw.rect(panel_surface, (0, 0, 0, 150), (0, 0, self.screen_width, 100))
            self.screen.blit(panel_surface, (0, 0))
            
            # 使用系统默认字体，避免中文显示问题
            font_large = pygame.font.Font(None, 36)
            font_medium = pygame.font.Font(None, 28)
            font_small = pygame.font.Font(None, 24)
            
            obs = self._get_obs()
            cart_pos, cart_vel, pole1_angle, pole1_angular_vel, pole2_angle, pole2_angular_vel = obs
            pole1_angle_deg = np.degrees(pole1_angle)
            pole2_angle_deg = np.degrees(pole2_angle)
            
            # 标题（使用英文）
            title_text = font_large.render("Double Pendulum Control", True, (255, 255, 255))
            self.screen.blit(title_text, (20, 10))
            
            # 第一摆杆角度
            pole1_color = (100, 255, 100) if abs(pole1_angle_deg) < 8 else \
                         (255, 200, 100) if abs(pole1_angle_deg) < 15 else (255, 100, 100)
            pole1_text = font_medium.render(f"Pole1: {pole1_angle_deg:.1f}deg", True, pole1_color)
            self.screen.blit(pole1_text, (20, 45))
            
            # 第二摆杆角度
            pole2_color = (100, 255, 100) if abs(pole2_angle_deg) < 10 else \
                         (255, 200, 100) if abs(pole2_angle_deg) < 20 else (255, 100, 100)
            pole2_text = font_medium.render(f"Pole2: {pole2_angle_deg:.1f}deg", True, pole2_color)
            self.screen.blit(pole2_text, (200, 45))
            
            # 位置显示
            pos_color = (100, 255, 100) if abs(cart_pos) < 1.5 else \
                       (255, 200, 100) if abs(cart_pos) < 2.5 else (255, 100, 100)
            pos_text = font_medium.render(f"Pos: {cart_pos:.2f}m", True, pos_color)
            self.screen.blit(pos_text, (380, 45))
            
            # 步数
            step_text = font_medium.render(f"Step: {self.current_step}", True, (200, 200, 255))
            self.screen.blit(step_text, (520, 45))
            
            # 状态指示
            if abs(pole1_angle_deg) < 8 and abs(pole2_angle_deg) < 12 and abs(cart_pos) < 1.5:
                status_text = font_medium.render("BALANCED", True, (100, 255, 100))
            else:
                status_text = font_medium.render("BALANCING...", True, (255, 200, 100))
            self.screen.blit(status_text, (self.screen_width - 200, 45))
            
            # 角速度信息
            vel_text = font_small.render(f"AngVel1: {pole1_angular_vel:.2f}", True, (180, 180, 180))
            self.screen.blit(vel_text, (20, 70))
            vel2_text = font_small.render(f"AngVel2: {pole2_angular_vel:.2f}", True, (180, 180, 180))
            self.screen.blit(vel2_text, (200, 70))
            
            # 扰动指示（如果启用）
            if self.enable_disturbance:
                disturbance_text = font_small.render(f"Disturbance: ON (Count: {self.disturbance_counter})", True, (255, 150, 150))
                self.screen.blit(disturbance_text, (380, 70))
                
                if abs(self.last_disturbance) > 0.01:
                    last_dist_text = font_small.render(f"Last: {self.last_disturbance:.2f}N", True, (255, 200, 150))
                    self.screen.blit(last_dist_text, (600, 70))
            
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def set_disturbance(self, enable=True, force_range=None, probability=None, disturbance_type=None, pole_ratio=None):
        """动态设置扰动参数"""
        self.enable_disturbance = enable
        if force_range is not None:
            self.disturbance_force_range = force_range
        if probability is not None:
            self.disturbance_probability = probability
        if disturbance_type is not None:
            self.disturbance_type = disturbance_type
        # pole_ratio参数在二阶倒立摆中暂不使用，但保持接口一致性
        
        # 重置扰动计数器
        self.disturbance_counter = 0
        self.last_disturbance = 0.0
        
        print(f"🌪️ 二阶倒立摆扰动设置:")
        print(f"   启用: {enable}")
        print(f"   力范围: ±{self.disturbance_force_range}N")
        print(f"   概率: {self.disturbance_probability*100:.1f}%")
        print(f"   预期频率: 每{1/self.disturbance_probability:.0f}步一次扰动")
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None