"""
基础物理环境 - 所有环境的父类
"""
import gymnasium as gym
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from abc import ABC, abstractmethod


class BasePhysicsEnv(gym.Env, ABC):
    """
    基础物理环境类
    
    提供：
    - Pymunk物理引擎初始化
    - Pygame渲染初始化
    - 通用的step/reset接口
    - 子类只需实现具体的物理对象和奖励
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None, config=None):
        """
        初始化基础环境
        
        Args:
            render_mode: 渲染模式 ('human', 'rgb_array', None)
            config: 环境配置字典
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.config = config or {}
        
        # 物理引擎
        self.space = None
        self.dt = self.config.get('dt', 1.0/60.0)
        self.max_steps = self.config.get('max_steps', 500)
        self.current_step = 0
        
        # 渲染
        self.screen = None
        self.clock = None
        self.draw_options = None
        self.screen_width = self.config.get('screen_width', 800)
        self.screen_height = self.config.get('screen_height', 600)
        
        # 初始化物理空间
        self._init_physics()
        
        # 子类需要定义observation_space和action_space
    
    def _init_physics(self):
        """初始化物理引擎"""
        self.space = pymunk.Space()
        self.space.gravity = (0, -self.config.get('gravity', 9.81) * 100)  # Pymunk uses pixels
    
    @abstractmethod
    def _create_bodies(self):
        """创建物理对象（子类实现）"""
        pass
    
    @abstractmethod
    def _get_obs(self):
        """获取观察（子类实现）"""
        pass
    
    @abstractmethod
    def _compute_reward(self, action):
        """计算奖励（子类实现）"""
        pass
    
    @abstractmethod
    def _check_termination(self):
        """检查是否终止（子类实现）"""
        pass
    
    def step(self, action):
        """
        执行一步
        
        Args:
            action: 动作
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # 应用动作（子类实现）
        self._apply_action(action)
        
        # 物理仿真
        self.space.step(self.dt)
        
        # 获取观察
        obs = self._get_obs()
        
        # 计算奖励
        reward, info = self._compute_reward(action)
        
        # 检查终止
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    @abstractmethod
    def _apply_action(self, action):
        """应用动作（子类实现）"""
        pass
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 清空物理空间
        for body in self.space.bodies:
            self.space.remove(body)
        for shape in self.space.shapes:
            self.space.remove(shape)
        for constraint in self.space.constraints:
            self.space.remove(constraint)
        
        # 重新创建物理对象
        self._create_bodies()
        
        # 重置步数
        self.current_step = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def render(self):
        """渲染"""
        if self.render_mode is None:
            return None
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption(self.__class__.__name__)
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # 清屏
        self.screen.fill((255, 255, 255))
        
        # 绘制物理对象
        self.space.debug_draw(self.draw_options)
        
        # 自定义绘制（子类可覆盖）
        self._custom_render()
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def _custom_render(self):
        """自定义渲染（子类可覆盖）"""
        pass
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
