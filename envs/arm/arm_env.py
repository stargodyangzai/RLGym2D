"""
简单的2关节机械臂环境 - 原型验证
任务：控制机械臂末端到达目标位置
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame


class ArmEnv(gym.Env):
    """2关节平面机械臂环境"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, config=None, reward_config=None):
        super().__init__()
        
        # 加载配置
        if config is None:
            from config import ENV_CONFIG
            config = ENV_CONFIG
        if reward_config is None:
            from config import REWARD_CONFIG
            reward_config = REWARD_CONFIG
        
        # 初始化奖励管理器
        self._init_reward_manager(reward_config)
        
        # 环境参数
        self.render_mode = render_mode
        self.dt = config.get('dt', 1.0 / 60.0)
        self.max_steps = config.get('max_steps', 500)
        self.current_step = 0
        
        # 渲染比例（公制单位转像素）
        self.pixels_per_meter = config.get('pixels_per_meter', 100)
        
        # 机械臂参数（公制单位：米）
        self.link_lengths = config.get('link_lengths', [1.0, 0.8])
        self.link_mass = config.get('link_mass', 0.5)
        self.max_torque = config.get('max_torque', 200.0)
        
        # 运动控制参数
        self.angular_velocity_limit = config.get('angular_velocity_limit', 20.0)
        self.velocity_damping = config.get('velocity_damping', 0.9)
        self.angular_velocity_damping = config.get('angular_velocity_damping', 0.9)
        
        # 目标生成参数（公制单位：米）
        self.target_distance_min = config.get('target_distance_min', 0.5)
        self.target_distance_max = config.get('target_distance_max', 1.5)
        self.target_angle_min = config.get('target_angle_min', 0)
        self.target_angle_max = config.get('target_angle_max', 3.14159)
        
        # 奖励参数（保留用于向后兼容）
        self.distance_reward_scale = reward_config.get('distance_reward_scale', 100.0)
        self.success_reward = reward_config.get('success_reward', 10.0)
        self.success_threshold = reward_config.get('success_threshold', 0.1)  # 米
        
        # 物理参数（公制单位：m/s²）
        self.gravity = config.get('gravity', 9.81)
        
        # 动作空间：两个关节的扭矩 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # 观测空间：[joint1_angle, joint2_angle, joint1_vel, joint2_vel, 
        #           target_x, target_y, end_x, end_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        # 初始化物理空间
        self.space = None
        self.base = None
        self.links = []
        self.joints = []
        self.target_pos = np.array([150.0, 100.0])
        
        # 渲染相关
        self.screen = None
        self.clock = None
        self.draw_options = None
        
        if self.render_mode == "human":
            self._init_render()
    
    def _init_reward_manager(self, reward_config):
        """初始化奖励管理器"""
        from reward_functions import (
            create_default_reward_manager,
            create_smooth_reward_manager,
            create_efficient_reward_manager,
            create_advanced_reward_manager
        )
        
        mode = reward_config.get('mode', 'default')
        
        if mode == 'default':
            self.reward_manager = create_default_reward_manager(reward_config)
        elif mode == 'smooth':
            self.reward_manager = create_smooth_reward_manager(reward_config)
        elif mode == 'efficient':
            self.reward_manager = create_efficient_reward_manager(reward_config)
        elif mode == 'advanced':
            self.reward_manager = create_advanced_reward_manager(reward_config)
        elif mode == 'custom':
            # 用户可以自定义奖励组件
            self.reward_manager = create_default_reward_manager(reward_config)
            # TODO: 添加自定义组件的逻辑
        else:
            self.reward_manager = create_default_reward_manager(reward_config)
    
    def _init_render(self):
        """初始化渲染"""
        pygame.init()
        # 可调整大小的窗口
        self.screen = pygame.display.set_mode((800, 800), pygame.RESIZABLE)
        pygame.display.set_caption("2D Robotic Arm Simulator - RL Training")
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 视图参数
        self.camera_offset = [0, 0]
        self.zoom = 1.0
    
    def _create_arm(self):
        """创建机械臂"""
        # 基座位置（世界坐标中心）
        base_pos = (0, 0)
        
        # 创建固定基座
        self.base = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.base.position = base_pos
        
        # 创建连杆
        self.links = []
        self.joints = []
        
        prev_body = self.base
        current_pos = base_pos
        
        for i, length in enumerate(self.link_lengths):
            # 创建连杆刚体（质心在连杆中心）
            moment = pymunk.moment_for_segment(
                self.link_mass, (-length/2, 0), (length/2, 0), 5
            )
            link_body = pymunk.Body(self.link_mass, moment)
            
            # 第一个连杆从基座开始，第二个连杆从第一个连杆末端开始
            if i == 0:
                link_body.position = (base_pos[0] + length/2, base_pos[1])
            else:
                # 获取前一个连杆的末端位置
                prev_link = self.links[i-1]
                prev_length = self.link_lengths[i-1]
                prev_angle = prev_link.angle
                prev_end_x = prev_link.position.x + prev_length/2 * np.cos(prev_angle)
                prev_end_y = prev_link.position.y + prev_length/2 * np.sin(prev_angle)
                link_body.position = (prev_end_x + length/2, prev_end_y)
            
            # 创建连杆形状（从连杆中心向两端延伸）
            link_shape = pymunk.Segment(link_body, (-length/2, 0), (length/2, 0), 5)
            link_shape.friction = 0.3  # 空气摩擦
            # 不同颜色区分两个连杆
            if i == 0:
                link_shape.color = (100, 150, 200, 255)  # 蓝色 - 第一段
            else:
                link_shape.color = (200, 100, 100, 255)  # 红色 - 第二段
            
            # 添加阻尼以防止过度震荡
            link_body.angular_velocity_limit = self.angular_velocity_limit
            link_body.velocity_damping = self.velocity_damping
            link_body.angular_velocity_damping = self.angular_velocity_damping
            
            self.space.add(link_body, link_shape)
            self.links.append(link_body)
            
            # 创建旋转关节（连接点在连杆的起始端）
            if i == 0:
                # 第一个关节连接基座和第一个连杆
                joint_pos = base_pos
                joint = pymunk.PivotJoint(prev_body, link_body, joint_pos)
            else:
                # 第二个关节连接第一个连杆末端和第二个连杆起始端
                prev_link = self.links[i-1]
                prev_length = self.link_lengths[i-1]
                # 关节位置在前一个连杆的局部坐标系中
                joint = pymunk.PivotJoint(
                    prev_link, link_body,
                    (prev_length/2, 0),  # 前一个连杆的末端（局部坐标）
                    (-length/2, 0)       # 当前连杆的起始端（局部坐标）
                )
            
            joint.collide_bodies = False
            self.space.add(joint)
            self.joints.append(joint)
            
            prev_body = link_body
    
    def _get_end_effector_pos(self):
        """获取末端执行器位置"""
        if len(self.links) == 0:
            return np.array([0.0, 0.0])
        
        last_link = self.links[-1]
        length = self.link_lengths[-1]
        angle = last_link.angle
        
        # 末端位置 = 连杆中心 + 半长度向量
        end_x = last_link.position.x + (length/2) * np.cos(angle)
        end_y = last_link.position.y + (length/2) * np.sin(angle)
        
        return np.array([end_x, end_y])
    
    def _get_end_effector_velocity(self):
        """获取末端执行器线速度（米/秒）"""
        if len(self.links) == 0:
            return np.array([0.0, 0.0])
        
        last_link = self.links[-1]
        length = self.link_lengths[-1]
        angle = last_link.angle
        angular_vel = last_link.angular_velocity
        
        # 末端线速度 = 连杆中心速度 + 旋转产生的速度
        # v_end = v_center + ω × r
        center_vx = last_link.velocity.x
        center_vy = last_link.velocity.y
        
        # 旋转产生的速度分量
        r_x = (length/2) * np.cos(angle)
        r_y = (length/2) * np.sin(angle)
        rot_vx = -angular_vel * r_y
        rot_vy = angular_vel * r_x
        
        end_vx = center_vx + rot_vx
        end_vy = center_vy + rot_vy
        
        return np.array([end_vx, end_vy])
    
    def _get_obs(self):
        """获取观测"""
        if len(self.links) == 0:
            return np.zeros(8, dtype=np.float32)
        
        # 关节角度和角速度
        angles = [link.angle for link in self.links]
        velocities = [link.angular_velocity for link in self.links]
        
        # 末端位置
        end_pos = self._get_end_effector_pos()
        
        # 归一化目标位置（相对于基座）
        target_rel = self.target_pos - np.array([0, 0])
        end_rel = end_pos - np.array([0, 0])
        
        obs = np.array([
            angles[0] / np.pi,
            angles[1] / np.pi,
            velocities[0] / 10.0,
            velocities[1] / 10.0,
            target_rel[0] / 200.0,
            target_rel[1] / 200.0,
            end_rel[0] / 200.0,
            end_rel[1] / 200.0,
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置奖励管理器
        self.reward_manager.reset()
        
        # 重新创建物理空间
        self.space = pymunk.Space()
        # 重力加速度 (向下为正，单位：m/s²)
        self.space.gravity = (0, self.gravity)
        
        # 创建机械臂
        self._create_arm()
        
        # 随机目标位置
        if seed is not None:
            np.random.seed(seed)
        
        angle = np.random.uniform(self.target_angle_min, self.target_angle_max)
        distance = np.random.uniform(self.target_distance_min, self.target_distance_max)
        self.target_pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """执行一步"""
        # 应用扭矩到关节
        action = np.clip(action, -1.0, 1.0)
        for i, link in enumerate(self.links):
            torque = action[i] * self.max_torque
            link.torque = torque
        
        # 物理仿真步进
        self.space.step(self.dt)
        
        # 获取观测
        obs = self._get_obs()
        
        # 准备环境状态用于奖励计算
        end_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(end_pos - self.target_pos)
        
        # 判断是否成功
        terminated = distance < self.success_threshold
        
        # 步数限制
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # 准备环境状态用于奖励计算（包含done状态）
        end_velocity = self._get_end_effector_velocity()
        env_state = {
            'distance': distance,
            'end_effector_pos': end_pos,
            'end_effector_velocity': end_velocity,  # 新增：末端速度
            'target_pos': self.target_pos,
            'joint_angles': [self.links[0].angle, self.links[1].angle],
            'joint_velocities': [self.links[0].angular_velocity, self.links[1].angular_velocity],
            'action': action,
            'current_step': self.current_step,
            'done': terminated or truncated,  # 回合是否结束
        }
        
        # 使用奖励管理器计算奖励
        reward, reward_info = self.reward_manager.compute_reward(env_state, log=False)
        
        # 渲染
        if self.render_mode == "human":
            self.render()
        
        # 返回额外信息
        info = {
            'distance': distance,
            'success': terminated,
            'reward_components': reward_info,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _world_to_screen(self, pos):
        """世界坐标转屏幕坐标（公制单位转像素）"""
        screen_w, screen_h = self.screen.get_size()
        # 将公制单位（米）转换为像素
        pos_pixels_x = pos[0] * self.pixels_per_meter
        pos_pixels_y = pos[1] * self.pixels_per_meter
        # 世界坐标原点在屏幕中心
        screen_x = screen_w / 2 + pos_pixels_x
        screen_y = screen_h / 2 - pos_pixels_y  # Y轴翻转
        return (int(screen_x), int(screen_y))
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
        
        if self.screen is None:
            self._init_render()
        
        # 获取当前窗口大小
        screen_w, screen_h = self.screen.get_size()
        
        # 渐变背景（深色主题）
        for y in range(screen_h):
            color_value = int(30 + (y / screen_h) * 20)
            pygame.draw.line(
                self.screen,
                (color_value, color_value, color_value + 10),
                (0, y),
                (screen_w, y)
            )
        
        # 绘制网格（0.5米间隔）
        grid_spacing = int(0.5 * self.pixels_per_meter)  # 0.5米 = 50像素
        grid_color = (60, 60, 70)
        for x in range(0, screen_w, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, screen_h), 1)
        for y in range(0, screen_h, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (screen_w, y), 1)
        
        # 绘制坐标轴
        center = (screen_w // 2, screen_h // 2)
        # X轴（红色）
        pygame.draw.line(self.screen, (200, 50, 50), 
                        (center[0] - 30, center[1]), 
                        (center[0] + 30, center[1]), 2)
        # Y轴（绿色）
        pygame.draw.line(self.screen, (50, 200, 50), 
                        (center[0], center[1] - 30), 
                        (center[0], center[1] + 30), 2)
        
        # 绘制基座（加强版）
        base_screen = self._world_to_screen((0, 0))
        # 基座底盘
        pygame.draw.circle(self.screen, (80, 80, 90), base_screen, 20)
        pygame.draw.circle(self.screen, (100, 100, 110), base_screen, 20, 3)
        # 基座中心
        pygame.draw.circle(self.screen, (120, 120, 130), base_screen, 8)
        pygame.draw.circle(self.screen, (60, 60, 70), base_screen, 8, 2)
        
        # 绘制机械臂连杆（美化版）
        for i, link in enumerate(self.links):
            length = self.link_lengths[i]
            angle = link.angle
            
            # 连杆起点和终点
            start_x = link.position.x - (length/2) * np.cos(angle)
            start_y = link.position.y - (length/2) * np.sin(angle)
            end_x = link.position.x + (length/2) * np.cos(angle)
            end_y = link.position.y + (length/2) * np.sin(angle)
            
            start_screen = self._world_to_screen((start_x, start_y))
            end_screen = self._world_to_screen((end_x, end_y))
            
            # 连杆颜色
            if i == 0:
                # 第一段：蓝色渐变
                color_main = (70, 130, 220)
                color_edge = (50, 100, 180)
            else:
                # 第二段：橙色渐变
                color_main = (255, 140, 60)
                color_edge = (220, 100, 40)
            
            # 绘制连杆阴影
            shadow_offset = 3
            pygame.draw.line(
                self.screen,
                (30, 30, 35),
                (start_screen[0] + shadow_offset, start_screen[1] + shadow_offset),
                (end_screen[0] + shadow_offset, end_screen[1] + shadow_offset),
                14
            )
            
            # 绘制连杆主体
            pygame.draw.line(self.screen, color_main, start_screen, end_screen, 12)
            # 绘制连杆边缘高光
            pygame.draw.line(self.screen, color_edge, start_screen, end_screen, 8)
            
            # 绘制关节
            joint_screen = self._world_to_screen((start_x, start_y))
            # 关节外圈
            pygame.draw.circle(self.screen, (40, 40, 50), joint_screen, 10)
            # 关节主体
            pygame.draw.circle(self.screen, (200, 200, 210), joint_screen, 8)
            # 关节高光
            pygame.draw.circle(self.screen, (240, 240, 250), joint_screen, 6)
            # 关节中心
            pygame.draw.circle(self.screen, (60, 60, 70), joint_screen, 3)
        
        # 绘制末端执行器
        end_pos = self._get_end_effector_pos()
        end_screen = self._world_to_screen(end_pos)
        # 末端阴影
        pygame.draw.circle(self.screen, (30, 30, 35), 
                          (end_screen[0] + 2, end_screen[1] + 2), 12)
        # 末端外圈
        pygame.draw.circle(self.screen, (255, 80, 80), end_screen, 10)
        # 末端主体
        pygame.draw.circle(self.screen, (255, 120, 120), end_screen, 8)
        # 末端高光
        pygame.draw.circle(self.screen, (255, 180, 180), end_screen, 5)
        
        # 绘制目标（简化版 - 绿色圆圈）
        target_screen = self._world_to_screen(self.target_pos)
        distance = np.linalg.norm(end_pos - self.target_pos)
        
        # 目标半径（公制单位转像素）
        target_radius = int(self.success_threshold * self.pixels_per_meter)
        
        # 绘制目标圆圈（绿色）
        pygame.draw.circle(self.screen, (0, 255, 100), target_screen, target_radius, 2)
        # 目标中心点
        pygame.draw.circle(self.screen, (0, 255, 100), target_screen, 3)
        
        # 绘制连接线（末端到目标）
        if distance > self.success_threshold:
            pygame.draw.line(self.screen, (100, 100, 120), 
                           end_screen, target_screen, 1)
        
        # 信息面板（半透明背景）
        panel_height = 120
        panel_surface = pygame.Surface((screen_w, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((20, 20, 25))
        self.screen.blit(panel_surface, (0, 0))
        
        # 显示信息（使用系统字体支持中文）
        try:
            # 尝试使用中文字体
            font_large = pygame.font.SysFont('notosanscjk,notosans,simsun,microsoftyahei,arial', 36)
            font_small = pygame.font.SysFont('notosanscjk,notosans,simsun,microsoftyahei,arial', 24)
        except:
            # 如果没有中文字体，使用默认字体和英文
            font_large = pygame.font.Font(None, 36)
            font_small = pygame.font.Font(None, 24)
        
        # 标题
        try:
            title = font_large.render("2D Robotic Arm RL", True, (200, 200, 220))
        except:
            title = font_large.render("2D Robotic Arm RL", True, (200, 200, 220))
        self.screen.blit(title, (20, 15))
        
        # 状态信息（使用英文避免字体问题）
        y_offset = 55
        success_status = "SUCCESS!" if distance < self.success_threshold else "Training..."
        info_lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Distance: {distance:.1f} px",
            f"Status: {success_status}"
        ]
        
        for i, line in enumerate(info_lines):
            if "SUCCESS" in line:
                color = (100, 255, 150)
            else:
                color = (180, 180, 200)
            text = font_small.render(line, True, color)
            self.screen.blit(text, (20, y_offset + i * 25))
        
        # 进度条
        progress = self.current_step / self.max_steps
        bar_width = 200
        bar_height = 8
        bar_x = screen_w - bar_width - 20
        bar_y = 20
        
        # 进度条背景
        pygame.draw.rect(self.screen, (50, 50, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        # 进度条填充
        pygame.draw.rect(self.screen, (100, 150, 255), 
                        (bar_x, bar_y, int(bar_width * progress), bar_height))
        # 进度条边框
        pygame.draw.rect(self.screen, (150, 150, 170), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
        
        # 帮助文本
        help_text = font_small.render("Tip: Window is resizable", True, (120, 120, 140))
        self.screen.blit(help_text, (screen_w - 220, screen_h - 30))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # 简单测试
    env = SimpleArmEnv(render_mode="human")
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
