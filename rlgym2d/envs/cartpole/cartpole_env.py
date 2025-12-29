"""
倒立摆环境（CartPole）

任务：控制小车左右移动，保持摆杆直立
- 观察：小车位置、速度、摆杆角度、角速度
- 动作：向左或向右施加力
- 目标：保持摆杆直立，小车在轨道中央
"""
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np


class CartPoleEnv(gym.Env):
    """
    倒立摆环境
    
    结构：
    - 小车（cart）：可以左右移动
    - 摆杆（pole）：通过关节连接到小车
    - 轨道（track）：限制小车移动范围
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, config=None):
        """
        初始化倒立摆环境
        
        Args:
            render_mode: 渲染模式
            config: 配置字典
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.config = config or {}
        
        # 物理参数
        self.dt = self.config.get('dt', 1.0/60.0)
        self.max_steps = self.config.get('max_steps', 500)
        self.gravity = self.config.get('gravity', 9.81)
        
        # 倒立摆参数
        self.cart_mass = 1.0  # kg
        self.pole_mass = 0.1  # kg
        self.pole_length = 1.0  # m (半长)
        self.force_mag = self.config.get('force_mag', 10.0)  # N
        
        # 轨道限制（从配置读取）
        self.track_length = self.config.get('position_threshold', 2.4) * 2  # 总轨道长度
        
        # 扰动配置
        self.enable_disturbance = self.config.get('enable_disturbance', False)
        self.disturbance_force_range = self.config.get('disturbance_force_range', 2.0)
        self.disturbance_probability = self.config.get('disturbance_probability', 0.02)
        
        # 扰动类型配置
        self.disturbance_type = self.config.get('disturbance_type', 'cart_only')
        self.pole_disturbance_ratio = self.config.get('pole_disturbance_ratio', 0.5)
        
        # 成功条件
        self.angle_threshold = self.config.get('angle_threshold', 12)  # 度
        self.position_threshold = self.config.get('position_threshold', 2.4)  # m
        
        # 物理引擎
        self.space = None
        self.cart = None
        self.pole = None
        self.joint = None
        
        # 渲染
        self.screen = None
        self.clock = None
        self.draw_options = None
        self.screen_width = 800
        self.screen_height = 600
        
        # 状态
        self.current_step = 0
        self.last_cart_disturbance = 0.0  # 记录最近的小车扰动
        self.last_pole_disturbance = 0.0  # 记录最近的摆杆扰动
        self.disturbance_counter = 0  # 扰动计数器
        
        # 观察空间：[cart_pos, cart_vel, pole_angle, pole_angular_vel]
        high = np.array([
            self.position_threshold * 2,
            np.finfo(np.float32).max,
            self.angle_threshold * 2 * np.pi / 180,
            np.finfo(np.float32).max
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
    
    def _create_cartpole(self):
        """创建倒立摆"""
        scale = 100  # 100像素 = 1米
        
        # 屏幕中心
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # 1. 创建小车
        cart_width = 0.5 * scale
        cart_height = 0.3 * scale
        cart_moment = pymunk.moment_for_box(self.cart_mass, (cart_width, cart_height))
        self.cart = pymunk.Body(self.cart_mass, cart_moment)
        self.cart.position = (center_x, center_y)
        
        cart_shape = pymunk.Poly.create_box(self.cart, (cart_width, cart_height))
        cart_shape.friction = 0.5
        cart_shape.color = (100, 150, 200, 255)
        self.space.add(self.cart, cart_shape)
        
        # 限制小车只能水平移动（移除PivotJoint，只保留GrooveJoint）
        groove_start = (center_x - self.track_length/2 * scale, center_y)
        groove_end = (center_x + self.track_length/2 * scale, center_y)
        groove = pymunk.GrooveJoint(self.space.static_body, self.cart, groove_start, groove_end, (0, 0))
        self.space.add(groove)
        
        # 2. 创建摆杆
        pole_length_px = self.pole_length * 2 * scale  # 总长度
        pole_moment = pymunk.moment_for_segment(
            self.pole_mass, (0, 0), (0, -pole_length_px), 5
        )
        self.pole = pymunk.Body(self.pole_mass, pole_moment)
        self.pole.position = (center_x, center_y - pole_length_px/2)
        
        pole_shape = pymunk.Segment(self.pole, (0, pole_length_px/2), (0, -pole_length_px/2), 5)
        pole_shape.friction = 0.5
        pole_shape.color = (200, 100, 100, 255)
        self.space.add(self.pole, pole_shape)
        
        # 3. 创建旋转关节（连接小车和摆杆）
        self.joint = pymunk.PivotJoint(self.cart, self.pole, (center_x, center_y))
        self.joint.collide_bodies = False
        self.space.add(self.joint)
    
    def _get_obs(self):
        """获取观察"""
        scale = 100
        center_x = self.screen_width // 2
        
        # 小车位置和速度
        cart_pos = (self.cart.position.x - center_x) / scale
        cart_vel = self.cart.velocity.x / scale
        
        # 摆杆角度和角速度（0度是向上）
        pole_angle = self.pole.angle
        pole_angular_vel = self.pole.angular_velocity
        
        obs = np.array([cart_pos, cart_vel, pole_angle, pole_angular_vel], dtype=np.float32)
        return obs
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 应用控制力到小车
        control_force = np.clip(action[0], -1.0, 1.0) * self.force_mag
        
        # 随机扰动（模拟外界干扰）
        cart_disturbance = 0.0
        pole_disturbance = 0.0
        
        if self.enable_disturbance and self.np_random is not None:
            if self.np_random.random() < self.disturbance_probability:
                # 生成基础扰动力
                base_disturbance = self.np_random.uniform(
                    -self.disturbance_force_range, 
                    self.disturbance_force_range
                )
                
                # 根据扰动类型分配力
                if self.disturbance_type == 'cart_only':
                    cart_disturbance = base_disturbance
                    pole_disturbance = 0.0
                elif self.disturbance_type == 'pole_only':
                    cart_disturbance = 0.0
                    pole_disturbance = base_disturbance
                elif self.disturbance_type == 'both':
                    cart_disturbance = base_disturbance
                    # 摆杆扰动可以是独立的随机值，也可以是相关的
                    pole_disturbance = base_disturbance * self.pole_disturbance_ratio
                    # 或者完全独立的随机扰动：
                    # pole_disturbance = self.np_random.uniform(
                    #     -self.disturbance_force_range * self.pole_disturbance_ratio,
                    #     self.disturbance_force_range * self.pole_disturbance_ratio
                    # )
                
                # 记录扰动
                self.last_cart_disturbance = cart_disturbance
                self.last_pole_disturbance = pole_disturbance
                self.disturbance_counter += 1
                
                # 打印扰动信息
                if self.disturbance_type == 'cart_only':
                    print(f"🚗 步骤 {self.current_step}: 小车扰动 {cart_disturbance:.2f}N (第{self.disturbance_counter}次)")
                elif self.disturbance_type == 'pole_only':
                    print(f"🎯 步骤 {self.current_step}: 摆杆扰动 {pole_disturbance:.2f}N (第{self.disturbance_counter}次)")
                elif self.disturbance_type == 'both':
                    print(f"🌪️ 步骤 {self.current_step}: 小车 {cart_disturbance:.2f}N + 摆杆 {pole_disturbance:.2f}N (第{self.disturbance_counter}次)")
        
        # 应用扰动力
        # 1. 小车扰动（水平方向）
        total_cart_force = control_force + cart_disturbance
        self.cart.apply_force_at_local_point((total_cart_force * 100, 0), (0, 0))
        
        # 2. 摆杆扰动（水平方向，作用在摆杆中部）
        if abs(pole_disturbance) > 0.001:
            pole_length_px = self.pole_length * 2 * 100  # 摆杆总长度（像素）
            # 在摆杆中部施加水平力
            self.pole.apply_force_at_local_point((pole_disturbance * 100, 0), (0, -pole_length_px/4))
        
        # 物理仿真
        self.space.step(self.dt)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward, info = self._compute_reward(obs)
        
        # 添加扰动信息到info
        info['cart_disturbance'] = float(cart_disturbance)
        info['pole_disturbance'] = float(pole_disturbance)
        info['control_force'] = float(control_force)
        info['total_cart_force'] = float(total_cart_force)
        info['disturbance_type'] = self.disturbance_type
        
        # 检查终止
        terminated = self._check_termination(obs)
        truncated = self.current_step >= self.max_steps
        
        # 成功标志
        info['success'] = truncated and not terminated
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs):
        """
        计算奖励
        
        设计理念：
        1. 主要奖励：保持摆杆直立，角度越大惩罚越重
        2. 次要奖励：保持小车在中央
        3. 惩罚：大的角速度（防止转圈）
        4. 存活奖励：基础奖励
        """
        cart_pos, cart_vel, pole_angle, pole_angular_vel = obs
        
        # 1. 角度奖励/惩罚（主要目标）
        # 使用二次惩罚，角度越大惩罚越重
        angle_deg = abs(np.degrees(pole_angle))
        if angle_deg < 5:
            # 很直立：高奖励
            angle_reward = 2.0
        elif angle_deg < 15:
            # 稍微倾斜：中等奖励
            angle_reward = 1.0 - (angle_deg - 5) * 0.1  # 从1.0线性下降到0
        else:
            # 倾斜太大：二次惩罚
            angle_penalty = -((angle_deg - 15) / 10) ** 2  # 二次增长的惩罚
            angle_reward = max(angle_penalty, -5.0)  # 限制最大惩罚
        
        # 2. 位置奖励（保持在中央）
        position_penalty = -abs(cart_pos) * 0.1
        
        # 3. 角速度惩罚（防止转圈，这是关键！）
        angular_vel_penalty = -abs(pole_angular_vel) * 0.2
        
        # 4. 小车速度惩罚（鼓励平稳）
        cart_vel_penalty = -abs(cart_vel) * 0.01
        
        # 5. 存活奖励（基础奖励）
        alive_reward = 1.0
        
        total_reward = angle_reward + position_penalty + angular_vel_penalty + cart_vel_penalty + alive_reward
        
        info = {
            'angle_reward': float(angle_reward),
            'position_penalty': float(position_penalty),
            'angular_vel_penalty': float(angular_vel_penalty),
            'cart_vel_penalty': float(cart_vel_penalty),
            'alive_reward': float(alive_reward),
            'cart_pos': float(cart_pos),
            'pole_angle_deg': float(angle_deg),
            'pole_angular_vel': float(pole_angular_vel),
            'reward_components': {
                'angle_reward': float(angle_reward),
                'position_penalty': float(position_penalty),
                'angular_vel_penalty': float(angular_vel_penalty),
                'cart_vel_penalty': float(cart_vel_penalty),
                'alive_reward': float(alive_reward),
            }
        }
        
        return total_reward, info
    
    def _check_termination(self, obs):
        """检查是否终止"""
        # 从配置中读取是否禁用终止条件
        disable_termination = self.config.get('disable_termination', False)
        
        if disable_termination:
            # 禁用终止条件，让智能体探索整个episode
            return False
        
        cart_pos, _, pole_angle, _ = obs
        
        # 小车超出轨道
        if abs(cart_pos) > self.position_threshold:
            return True
        
        # 摆杆倾斜太大
        angle_deg = abs(np.degrees(pole_angle))
        if angle_deg > self.angle_threshold:
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
        
        # 重新创建倒立摆
        self._create_cartpole()
        
        # 添加小的随机扰动
        if self.np_random is not None:
            self.pole.angle = self.np_random.uniform(-0.05, 0.05)
            self.cart.position = (
                self.cart.position.x + self.np_random.uniform(-10, 10),
                self.cart.position.y
            )
        
        # 重置步数和扰动记录
        self.current_step = 0
        self.last_cart_disturbance = 0.0
        self.last_pole_disturbance = 0.0
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
                pygame.display.set_caption("CartPole - Inverted Pendulum")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 背景渐变
        for y in range(self.screen_height):
            color_ratio = y / self.screen_height
            r = int(240 + (255 - 240) * color_ratio)
            g = int(248 + (255 - 248) * color_ratio)
            b = int(255)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.screen_width, y))
        
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
                        (center_x, center_y - 10), (center_x, center_y + 10), 2)
        
        # 绘制物理对象
        self.space.debug_draw(self.draw_options)
        
        # UI信息面板
        if self.render_mode == 'human':
            # 半透明背景
            panel_surface = pygame.Surface((self.screen_width, 80), pygame.SRCALPHA)
            pygame.draw.rect(panel_surface, (0, 0, 0, 150), (0, 0, self.screen_width, 80))
            self.screen.blit(panel_surface, (0, 0))
            
            font_large = pygame.font.Font(None, 42)
            font_medium = pygame.font.Font(None, 32)
            
            obs = self._get_obs()
            cart_pos, cart_vel, pole_angle, pole_angular_vel = obs
            angle_deg = np.degrees(pole_angle)
            
            # 角度显示（带颜色指示）
            angle_color = (100, 255, 100) if abs(angle_deg) < 5 else \
                         (255, 200, 100) if abs(angle_deg) < 10 else (255, 100, 100)
            angle_text = font_large.render(f"Angle: {angle_deg:.1f}deg", True, angle_color)
            self.screen.blit(angle_text, (20, 15))
            
            # 位置显示
            pos_color = (100, 255, 100) if abs(cart_pos) < 1.0 else \
                       (255, 200, 100) if abs(cart_pos) < 2.0 else (255, 100, 100)
            pos_text = font_medium.render(f"Pos: {cart_pos:.2f}m", True, pos_color)
            self.screen.blit(pos_text, (250, 20))
            
            # 步数
            step_text = font_medium.render(f"Step: {self.current_step}", True, (200, 200, 255))
            self.screen.blit(step_text, (450, 20))
            
            # 状态指示
            if abs(angle_deg) < 5 and abs(cart_pos) < 1.0:
                status_text = font_medium.render("BALANCED", True, (100, 255, 100))
            else:
                status_text = font_medium.render("BALANCING...", True, (255, 200, 100))
            self.screen.blit(status_text, (self.screen_width - 200, 20))
            
            # 扰动指示（如果启用）
            if self.enable_disturbance:
                disturbance_text = font_medium.render(f"DISTURBANCE: {self.disturbance_type.upper()} (Count: {self.disturbance_counter})", True, (255, 150, 150))
                self.screen.blit(disturbance_text, (20, 50))
                
                # 显示最近的扰动
                if self.disturbance_type == 'cart_only' and abs(self.last_cart_disturbance) > 0.01:
                    last_dist_text = font_medium.render(f"Cart: {self.last_cart_disturbance:.2f}N", True, (255, 200, 150))
                    self.screen.blit(last_dist_text, (400, 50))
                elif self.disturbance_type == 'pole_only' and abs(self.last_pole_disturbance) > 0.01:
                    last_dist_text = font_medium.render(f"Pole: {self.last_pole_disturbance:.2f}N", True, (255, 200, 150))
                    self.screen.blit(last_dist_text, (400, 50))
                elif self.disturbance_type == 'both' and (abs(self.last_cart_disturbance) > 0.01 or abs(self.last_pole_disturbance) > 0.01):
                    last_dist_text = font_medium.render(f"C:{self.last_cart_disturbance:.1f}N P:{self.last_pole_disturbance:.1f}N", True, (255, 200, 150))
                    self.screen.blit(last_dist_text, (400, 50))
            
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def set_disturbance(self, enable=True, force_range=None, probability=None, disturbance_type=None, pole_ratio=None):
        """动态设置扰动参数（用于演示时测试鲁棒性）"""
        self.enable_disturbance = enable
        if force_range is not None:
            self.disturbance_force_range = force_range
        if probability is not None:
            self.disturbance_probability = probability
        if disturbance_type is not None:
            self.disturbance_type = disturbance_type
        if pole_ratio is not None:
            self.pole_disturbance_ratio = pole_ratio
        
        # 重置扰动计数器
        self.disturbance_counter = 0
        self.last_cart_disturbance = 0.0
        self.last_pole_disturbance = 0.0
        
        print(f"🌪️ 扰动设置更新:")
        print(f"   启用: {enable}")
        print(f"   类型: {self.disturbance_type}")
        print(f"   力范围: ±{self.disturbance_force_range}N")
        print(f"   概率: {self.disturbance_probability*100:.1f}%")
        if self.disturbance_type == 'both':
            print(f"   摆杆比例: {self.pole_disturbance_ratio}")
        print(f"   预期频率: 每{1/self.disturbance_probability:.0f}步一次扰动")
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
