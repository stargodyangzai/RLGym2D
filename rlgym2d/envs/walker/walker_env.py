"""
2D火柴人行走环境

任务：训练一个2D火柴人学会行走
- 观察：关节角度、角速度、身体位置、速度、倾斜角度
- 动作：各关节的扭矩
- 目标：向右行走，保持平衡，不摔倒
"""
import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np


class WalkerEnv(gym.Env):
    """
    2D火柴人行走环境
    
    火柴人结构：
    - 躯干（torso）
    - 大腿（thigh）× 2
    - 小腿（calf）× 2
    - 共5个刚体，4个关节
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, config=None):
        """
        初始化火柴人环境
        
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
        
        # 火柴人参数（单位：米）
        self.torso_length = 0.6
        self.thigh_length = 0.4
        self.calf_length = 0.4
        self.body_mass = 10.0
        self.leg_mass = 5.0
        
        # 关节限制（弧度）
        self.hip_limit = np.pi / 3  # ±60度
        self.knee_limit = np.pi / 2  # 膝关节弯曲范围
        
        # 控制参数
        self.max_torque = self.config.get('max_torque', 100.0)
        
        # 物理引擎
        self.space = None
        self.ground = None
        self.torso = None
        self.left_thigh = None
        self.right_thigh = None
        self.left_calf = None
        self.right_calf = None
        self.joints = []
        
        # 渲染
        self.screen = None
        self.clock = None
        self.draw_options = None
        self.screen_width = 1200
        self.screen_height = 600
        self.camera_x = 0  # 相机跟随
        
        # 状态
        self.current_step = 0
        self.initial_x = 0
        
        # 观察空间：[关节角度×4, 关节角速度×4, 躯干位置×2, 躯干速度×2, 躯干角度, 躯干角速度]
        # 总共：4 + 4 + 2 + 2 + 1 + 1 = 14维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        
        # 动作空间：4个关节的扭矩 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # 初始化物理引擎
        self._init_physics()
    
    def _init_physics(self):
        """初始化物理引擎"""
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity * 100)  # 正值向下，Pymunk uses pixels
        
        # 创建地面（在屏幕下方）
        ground_body = self.space.static_body
        ground_y = 550  # 地面在屏幕底部附近
        ground_shape = pymunk.Segment(ground_body, (-10000, ground_y), (10000, ground_y), 5)
        ground_shape.friction = 0.8
        self.space.add(ground_shape)
        self.ground = ground_shape
    
    def _create_walker(self):
        """创建火柴人 - 使用与机械臂相同的方式"""
        # 像素比例：100像素 = 1米
        scale = 100
        
        # 起始位置（髋部位置）
        hip_x = 300
        # 计算合适的起始高度：
        # 地面=550，脚底需要在地面上
        # 脚底 = 髋部 + 大腿(40) + 小腿(40) = 髋部 + 80
        # 所以髋部 = 地面 - 80 - 10(安全间隙) = 460
        hip_y = 460  # 之前是450，太低了
        
        # 计算各部分长度
        torso_length_px = self.torso_length * scale  # 60px
        thigh_length_px = self.thigh_length * scale  # 40px
        calf_length_px = self.calf_length * scale    # 40px
        
        # 1. 创建躯干（从髋部向上）
        torso_mass = self.body_mass
        # Segment从质心向两端延伸
        torso_moment = pymunk.moment_for_segment(torso_mass, (0, -torso_length_px/2), (0, torso_length_px/2), 5)
        self.torso = pymunk.Body(torso_mass, torso_moment)
        # 质心位置 = 髋部 - 半个躯干长度（向上）
        self.torso.position = (hip_x, hip_y - torso_length_px/2)
        torso_shape = pymunk.Segment(self.torso, (0, -torso_length_px/2), (0, torso_length_px/2), 5)
        torso_shape.friction = 0.5
        torso_shape.color = (200, 100, 100, 255)
        self.space.add(self.torso, torso_shape)
        
        # 2. 创建左大腿（从髋部向下）
        thigh_mass = self.leg_mass
        thigh_moment = pymunk.moment_for_segment(thigh_mass, (0, -thigh_length_px/2), (0, thigh_length_px/2), 5)
        self.left_thigh = pymunk.Body(thigh_mass, thigh_moment)
        # 质心位置 = 髋部 + 半个大腿长度（向下）
        self.left_thigh.position = (hip_x, hip_y + thigh_length_px/2)
        left_thigh_shape = pymunk.Segment(self.left_thigh, (0, -thigh_length_px/2), (0, thigh_length_px/2), 5)
        left_thigh_shape.friction = 0.5
        left_thigh_shape.color = (100, 150, 200, 255)
        self.space.add(self.left_thigh, left_thigh_shape)
        
        # 3. 创建右大腿（从髋部向下）
        self.right_thigh = pymunk.Body(thigh_mass, thigh_moment)
        self.right_thigh.position = (hip_x, hip_y + thigh_length_px/2)
        right_thigh_shape = pymunk.Segment(self.right_thigh, (0, -thigh_length_px/2), (0, thigh_length_px/2), 5)
        right_thigh_shape.friction = 0.5
        right_thigh_shape.color = (100, 150, 200, 255)
        self.space.add(self.right_thigh, right_thigh_shape)
        
        # 4. 创建左小腿（从膝关节向下）
        calf_mass = self.leg_mass * 0.8
        calf_moment = pymunk.moment_for_segment(calf_mass, (0, -calf_length_px/2), (0, calf_length_px/2), 5)
        self.left_calf = pymunk.Body(calf_mass, calf_moment)
        # 膝关节位置 = 髋部 + 大腿长度
        knee_y = hip_y + thigh_length_px
        # 质心位置 = 膝关节 + 半个小腿长度（向下）
        self.left_calf.position = (hip_x, knee_y + calf_length_px/2)
        left_calf_shape = pymunk.Segment(self.left_calf, (0, -calf_length_px/2), (0, calf_length_px/2), 5)
        left_calf_shape.friction = 0.8
        left_calf_shape.color = (100, 200, 150, 255)
        self.space.add(self.left_calf, left_calf_shape)
        
        # 5. 创建右小腿（从膝关节向下）
        self.right_calf = pymunk.Body(calf_mass, calf_moment)
        self.right_calf.position = (hip_x, knee_y + calf_length_px/2)
        right_calf_shape = pymunk.Segment(self.right_calf, (0, -calf_length_px/2), (0, calf_length_px/2), 5)
        right_calf_shape.friction = 0.8
        right_calf_shape.color = (100, 200, 150, 255)
        self.space.add(self.right_calf, right_calf_shape)
        
        # 6. 创建关节（完全模仿机械臂的方式）
        self.joints = []
        
        # 左髋关节 - 使用世界坐标（像机械臂的第一个关节）
        left_hip = pymunk.PivotJoint(self.torso, self.left_thigh, (hip_x, hip_y))
        left_hip.collide_bodies = False
        left_hip_motor = pymunk.SimpleMotor(self.torso, self.left_thigh, 0)
        left_hip_limit = pymunk.RotaryLimitJoint(self.torso, self.left_thigh, -self.hip_limit, self.hip_limit)
        self.space.add(left_hip, left_hip_motor, left_hip_limit)
        self.joints.append(('left_hip', left_hip_motor))
        
        # 右髋关节 - 使用世界坐标
        right_hip = pymunk.PivotJoint(self.torso, self.right_thigh, (hip_x, hip_y))
        right_hip.collide_bodies = False
        right_hip_motor = pymunk.SimpleMotor(self.torso, self.right_thigh, 0)
        right_hip_limit = pymunk.RotaryLimitJoint(self.torso, self.right_thigh, -self.hip_limit, self.hip_limit)
        self.space.add(right_hip, right_hip_motor, right_hip_limit)
        self.joints.append(('right_hip', right_hip_motor))
        
        # 左膝关节 - 使用局部坐标（像机械臂的后续关节）
        left_knee = pymunk.PivotJoint(
            self.left_thigh, self.left_calf,
            (0, thigh_length_px/2),   # 大腿的末端（局部坐标）
            (0, -calf_length_px/2)    # 小腿的起始端（局部坐标）
        )
        left_knee.collide_bodies = False
        left_knee_motor = pymunk.SimpleMotor(self.left_thigh, self.left_calf, 0)
        left_knee_limit = pymunk.RotaryLimitJoint(self.left_thigh, self.left_calf, -self.knee_limit, 0)
        self.space.add(left_knee, left_knee_motor, left_knee_limit)
        self.joints.append(('left_knee', left_knee_motor))
        
        # 右膝关节 - 使用局部坐标
        right_knee = pymunk.PivotJoint(
            self.right_thigh, self.right_calf,
            (0, thigh_length_px/2),   # 大腿的末端（局部坐标）
            (0, -calf_length_px/2)    # 小腿的起始端（局部坐标）
        )
        right_knee.collide_bodies = False
        right_knee_motor = pymunk.SimpleMotor(self.right_thigh, self.right_calf, 0)
        right_knee_limit = pymunk.RotaryLimitJoint(self.right_thigh, self.right_calf, -self.knee_limit, 0)
        self.space.add(right_knee, right_knee_motor, right_knee_limit)
        self.joints.append(('right_knee', right_knee_motor))
        
        # 记录初始位置
        self.initial_x = self.torso.position.x
    
    def _get_obs(self):
        """获取观察"""
        # 关节角度（相对角度）
        left_hip_angle = self.left_thigh.angle - self.torso.angle
        right_hip_angle = self.right_thigh.angle - self.torso.angle
        left_knee_angle = self.left_calf.angle - self.left_thigh.angle
        right_knee_angle = self.right_calf.angle - self.right_thigh.angle
        
        # 关节角速度
        left_hip_vel = self.left_thigh.angular_velocity - self.torso.angular_velocity
        right_hip_vel = self.right_thigh.angular_velocity - self.torso.angular_velocity
        left_knee_vel = self.left_calf.angular_velocity - self.left_thigh.angular_velocity
        right_knee_vel = self.right_calf.angular_velocity - self.right_thigh.angular_velocity
        
        # 躯干状态
        torso_x, torso_y = self.torso.position
        torso_vx, torso_vy = self.torso.velocity
        torso_angle = self.torso.angle
        torso_angular_vel = self.torso.angular_velocity
        
        obs = np.array([
            left_hip_angle / np.pi,
            right_hip_angle / np.pi,
            left_knee_angle / np.pi,
            right_knee_angle / np.pi,
            left_hip_vel / 10.0,
            right_hip_vel / 10.0,
            left_knee_vel / 10.0,
            right_knee_vel / 10.0,
            (torso_x - self.initial_x) / 1000.0,
            torso_y / 1000.0,
            torso_vx / 100.0,
            torso_vy / 100.0,
            torso_angle / np.pi,
            torso_angular_vel / 10.0,
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 应用动作（扭矩）
        action = np.clip(action, -1.0, 1.0)
        for i, (name, motor) in enumerate(self.joints):
            torque = action[i] * self.max_torque
            motor.max_force = abs(torque)
            motor.rate = np.sign(torque) * 10  # 控制方向
        
        # 物理仿真
        self.space.step(self.dt)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward, info = self._compute_reward(action)
        
        # 检查终止（摔倒）
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # 添加成功标志：只有走得足够远才算成功
        # 摔倒不算成功，只有完成max_steps才可能成功
        distance = (self.torso.position.x - self.initial_x) / 100.0
        success = (not terminated) and truncated and (distance > 5.0)  # 走超过5米算成功
        info['success'] = success
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action):
        """
        计算奖励 - 参考OpenAI Gym BipedalWalker和相关研究
        
        设计理念：
        1. 主要奖励前进速度（而不是距离）
        2. 惩罚能量消耗（动作幅度）
        3. 轻微的存活奖励
        4. 惩罚不自然的姿态
        """
        # 获取当前状态
        torso_x, torso_y = self.torso.position
        torso_vx, torso_vy = self.torso.velocity
        torso_angle = self.torso.angle
        
        # 1. 前进速度奖励（主要驱动力）
        # 目标速度约为 1-2 m/s，过快或过慢都不好
        target_velocity = 1.5  # m/s
        current_velocity = torso_vx / 100.0  # 转换为 m/s
        # 使用高斯奖励，在目标速度附近最高
        velocity_reward = np.exp(-((current_velocity - target_velocity) ** 2) / 0.5) * 2.0
        # 如果向后走，给予惩罚
        if current_velocity < 0:
            velocity_reward = current_velocity * 2.0
        
        # 2. 能量消耗惩罚（鼓励高效运动）
        # 参考 BipedalWalker: -0.00035 * action^2
        action_penalty = -np.sum(np.square(action)) * 0.001
        
        # 3. 躯干直立奖励（保持平衡）
        # 躯干应该接近垂直
        upright_reward = np.cos(torso_angle) * 0.5
        
        # 4. 躯干高度奖励（保持站立）
        # 理想高度：髋部460 - 躯干一半30 = 430
        target_height = 430
        height_diff = abs(torso_y - target_height)
        height_reward = -height_diff / 100.0 * 0.3
        
        # 5. 躯干垂直速度惩罚（减少跳跃）
        vertical_velocity_penalty = -abs(torso_vy / 100.0) * 0.1
        
        # 6. 关节角度惩罚（避免不自然的姿态）
        # 获取关节角度
        left_hip_angle = self.left_thigh.angle - self.torso.angle
        right_hip_angle = self.right_thigh.angle - self.torso.angle
        left_knee_angle = self.left_calf.angle - self.left_thigh.angle
        right_knee_angle = self.right_calf.angle - self.right_thigh.angle
        
        # 惩罚极端的关节角度
        joint_angles = [left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle]
        joint_penalty = -sum([abs(angle) for angle in joint_angles if abs(angle) > np.pi/4]) * 0.1
        
        # 7. 关节角速度惩罚（鼓励平滑运动）
        left_hip_vel = self.left_thigh.angular_velocity - self.torso.angular_velocity
        right_hip_vel = self.right_thigh.angular_velocity - self.torso.angular_velocity
        left_knee_vel = self.left_calf.angular_velocity - self.left_thigh.angular_velocity
        right_knee_vel = self.right_calf.angular_velocity - self.right_thigh.angular_velocity
        
        angular_velocity_penalty = -(abs(left_hip_vel) + abs(right_hip_vel) + 
                                     abs(left_knee_vel) + abs(right_knee_vel)) / 40.0 * 0.05
        
        # 8. 脚部接触地面奖励（鼓励稳定步态）
        # 检查脚是否接近地面
        left_foot_y = self.left_calf.position.y + 20  # 小腿底部
        right_foot_y = self.right_calf.position.y + 20
        ground_y = 550
        
        left_foot_contact = 1.0 if abs(left_foot_y - ground_y) < 10 else 0.0
        right_foot_contact = 1.0 if abs(right_foot_y - ground_y) < 10 else 0.0
        # 至少一只脚接触地面
        foot_contact_reward = max(left_foot_contact, right_foot_contact) * 0.2
        
        # 总奖励
        total_reward = (
            velocity_reward +
            action_penalty +
            upright_reward +
            height_reward +
            vertical_velocity_penalty +
            joint_penalty +
            angular_velocity_penalty +
            foot_contact_reward
        )
        
        # 记录详细信息
        info = {
            'velocity_reward': float(velocity_reward),
            'action_penalty': float(action_penalty),
            'upright_reward': float(upright_reward),
            'height_reward': float(height_reward),
            'vertical_velocity_penalty': float(vertical_velocity_penalty),
            'joint_penalty': float(joint_penalty),
            'angular_velocity_penalty': float(angular_velocity_penalty),
            'foot_contact_reward': float(foot_contact_reward),
            'distance': float((torso_x - self.initial_x) / 100.0),
            'velocity': float(current_velocity),
            'reward_components': {
                'velocity_reward': float(velocity_reward),
                'action_penalty': float(action_penalty),
                'upright_reward': float(upright_reward),
                'height_reward': float(height_reward),
                'vertical_velocity_penalty': float(vertical_velocity_penalty),
                'joint_penalty': float(joint_penalty),
                'angular_velocity_penalty': float(angular_velocity_penalty),
                'foot_contact_reward': float(foot_contact_reward),
            }
        }
        
        return total_reward, info
    
    def _check_termination(self):
        """
        检查是否终止（摔倒）
        
        注意：初期训练时可以禁用终止条件，让火柴人充分探索
        设置 DISABLE_TERMINATION=True 来禁用
        """
        # 从配置中读取是否禁用终止条件
        disable_termination = self.config.get('disable_termination', False)
        
        if disable_termination:
            # 禁用终止条件，让火柴人探索整个episode
            return False
        
        # 躯干高度太低（摔倒到地面）
        # 理想高度约430，允许下降到510（接近地面）
        if self.torso.position.y > 510:  # 地面是550
            return True
        
        # 躯干倾斜太大（失去平衡）
        # 放宽到90度
        if abs(self.torso.angle) > np.pi / 2:  # 90度
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 清空物理空间（保留地面）
        for body in list(self.space.bodies):
            self.space.remove(body)
        for shape in list(self.space.shapes):
            if shape != self.ground:
                self.space.remove(shape)
        for constraint in list(self.space.constraints):
            self.space.remove(constraint)
        
        # 重新创建火柴人
        self._create_walker()
        
        # 重置步数
        self.current_step = 0
        
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
                pygame.display.set_caption("2D Walker - Reinforcement Learning Demo")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 相机跟随
        self.camera_x = self.torso.position.x - self.screen_width // 2
        
        # ========== 背景渐变 ==========
        # 天空渐变（从浅蓝到白色）
        for y in range(0, 550):
            color_ratio = y / 550
            r = int(135 + (255 - 135) * color_ratio)
            g = int(206 + (255 - 206) * color_ratio)
            b = int(235 + (255 - 235) * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.screen_width, y))
        
        # 地面（棕色）
        pygame.draw.rect(self.screen, (139, 90, 60), (0, 550, self.screen_width, 50))
        
        # ========== 装饰元素 ==========
        # 1. 绘制云朵（固定位置，不跟随相机）
        cloud_color = (255, 255, 255, 180)
        clouds = [(200, 100), (500, 150), (800, 80), (1000, 120)]
        for cx, cy in clouds:
            # 简单的云朵形状
            pygame.draw.circle(self.screen, cloud_color, (cx, cy), 30)
            pygame.draw.circle(self.screen, cloud_color, (cx + 25, cy), 25)
            pygame.draw.circle(self.screen, cloud_color, (cx - 25, cy), 25)
            pygame.draw.circle(self.screen, cloud_color, (cx + 15, cy - 15), 20)
        
        # ========== 参考网格（精细化） ==========
        # 2. 绘制精细网格线（每100像素=1米）
        grid_color_light = (220, 235, 245)  # 浅蓝灰色
        grid_color_dark = (180, 200, 220)   # 深蓝灰色
        grid_spacing = 100  # 1米
        
        # 计算可见范围
        start_x = int(self.camera_x // grid_spacing) * grid_spacing
        end_x = start_x + self.screen_width + grid_spacing
        
        # 绘制垂直网格线
        for i, x in enumerate(range(start_x, end_x, grid_spacing)):
            screen_x = x - self.camera_x
            if 0 <= screen_x <= self.screen_width:
                # 每5米用深色线
                color = grid_color_dark if i % 5 == 0 else grid_color_light
                width = 2 if i % 5 == 0 else 1
                pygame.draw.line(self.screen, color, (screen_x, 0), (screen_x, 550), width)
        
        # 绘制水平网格线
        for i, y in enumerate(range(0, 550, grid_spacing)):
            color = grid_color_dark if i % 5 == 0 else grid_color_light
            width = 2 if i % 5 == 0 else 1
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y), width)
        
        # ========== 距离标记（美化） ==========
        marker_spacing = 500  # 5米
        font_small = pygame.font.Font(None, 28)
        
        start_marker = int(self.initial_x // marker_spacing) * marker_spacing
        end_marker = int((self.camera_x + self.screen_width) // marker_spacing + 1) * marker_spacing
        
        for x in range(start_marker, end_marker, marker_spacing):
            screen_x = x - self.camera_x
            if 0 <= screen_x <= self.screen_width:
                # 绘制标记柱
                pygame.draw.rect(self.screen, (100, 100, 100), (screen_x - 3, 530, 6, 20))
                # 绘制距离文字（带阴影）
                distance_m = (x - self.initial_x) / 100.0
                text_str = f"{distance_m:.0f}m"
                # 阴影
                text_shadow = font_small.render(text_str, True, (50, 50, 50))
                self.screen.blit(text_shadow, (screen_x - 18, 557))
                # 主文字
                text = font_small.render(text_str, True, (255, 255, 255))
                self.screen.blit(text, (screen_x - 20, 555))
        
        # ========== 起点标记（美化） ==========
        start_screen_x = self.initial_x - self.camera_x
        if -50 <= start_screen_x <= self.screen_width + 50:
            # 绿色旗帜
            pygame.draw.line(self.screen, (34, 139, 34), (start_screen_x, 530), (start_screen_x, 480), 3)
            # 旗帜三角形
            flag_points = [(start_screen_x, 480), (start_screen_x + 30, 490), (start_screen_x, 500)]
            pygame.draw.polygon(self.screen, (50, 205, 50), flag_points)
            # START文字
            font_start = pygame.font.Font(None, 32)
            start_text = font_start.render("START", True, (255, 255, 255))
            # 绿色背景
            text_rect = start_text.get_rect()
            text_rect.center = (start_screen_x, 460)
            pygame.draw.rect(self.screen, (34, 139, 34), text_rect.inflate(10, 5))
            self.screen.blit(start_text, text_rect)
        
        # ========== 绘制物理对象（使用相机变换） ==========
        self.draw_options.transform = pymunk.Transform.translation(-self.camera_x, 0)
        self.space.debug_draw(self.draw_options)
        
        # ========== UI信息面板（美化） ==========
        if self.render_mode == 'human':
            # 半透明背景面板
            panel_surface = pygame.Surface((self.screen_width, 100), pygame.SRCALPHA)
            pygame.draw.rect(panel_surface, (0, 0, 0, 150), (0, 0, self.screen_width, 100))
            self.screen.blit(panel_surface, (0, 0))
            
            # 主要信息
            font_large = pygame.font.Font(None, 42)
            font_medium = pygame.font.Font(None, 32)
            font_small = pygame.font.Font(None, 24)
            
            distance = self.torso.position.x - self.initial_x
            velocity = self.torso.velocity.x / 100.0
            
            # 距离（大字，带图标）
            distance_text = font_large.render(f"🏃 {distance/100:.2f}m", True, (255, 255, 255))
            self.screen.blit(distance_text, (20, 15))
            
            # 速度
            speed_color = (100, 255, 100) if velocity > 0 else (255, 100, 100)
            speed_text = font_medium.render(f"⚡ {velocity:.2f}m/s", True, speed_color)
            self.screen.blit(speed_text, (20, 55))
            
            # 步数
            step_text = font_medium.render(f"👣 {self.current_step}", True, (200, 200, 255))
            self.screen.blit(step_text, (250, 55))
            
            # 右侧信息
            height = self.torso.position.y
            angle = np.degrees(self.torso.angle)
            
            # 高度
            height_text = font_small.render(f"Height: {height:.0f}px", True, (200, 200, 200))
            self.screen.blit(height_text, (self.screen_width - 200, 20))
            
            # 角度（带颜色指示）
            angle_color = (100, 255, 100) if abs(angle) < 30 else (255, 200, 100) if abs(angle) < 60 else (255, 100, 100)
            angle_text = font_small.render(f"Angle: {angle:.1f}°", True, angle_color)
            self.screen.blit(angle_text, (self.screen_width - 200, 45))
            
            # 进度条（距离）
            max_distance = 50.0  # 假设最大50米
            progress = min(distance / 100.0 / max_distance, 1.0)
            bar_width = 300
            bar_height = 15
            bar_x = self.screen_width - bar_width - 20
            bar_y = 70
            
            # 进度条背景
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=7)
            # 进度条填充
            if progress > 0:
                fill_width = int(bar_width * progress)
                color = (100, 255, 100) if progress < 0.5 else (255, 200, 100) if progress < 0.8 else (255, 100, 100)
                pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height), border_radius=7)
            
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
