"""
奖励函数模块 - 可扩展的奖励计算系统

设计理念：
1. 每个奖励组件是独立的函数
2. 可以灵活组合多个奖励
3. 每个奖励有独立的权重
4. 便于调试和分析
"""
import numpy as np  # 导入numpy用于数学计算
from typing import Dict, Callable, Any  # 导入类型提示


class RewardFunction:
    """奖励函数基类 - 所有奖励组件的父类"""
    
    def __init__(self, weight: float = 1.0):
        """
        初始化奖励函数
        
        Args:
            weight: 奖励权重，用于调整该组件在总奖励中的重要性
        """
        self.weight = weight  # 保存权重
        self.name = self.__class__.__name__  # 自动获取类名作为组件名称
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算奖励值（子类必须实现）
        
        Args:
            env_state: 环境状态字典，包含距离、位置、速度等信息
            
        Returns:
            float: 该组件的奖励值
        """
        raise NotImplementedError  # 抽象方法，子类必须实现
    
    def __call__(self, env_state: Dict[str, Any]) -> float:
        """
        调用奖励函数（使对象可以像函数一样调用）
        
        Args:
            env_state: 环境状态字典
            
        Returns:
            float: 加权后的奖励值 = weight × compute(env_state)
        """
        return self.weight * self.compute(env_state)  # 返回加权奖励


# ============================================================================
# 基础奖励组件
# ============================================================================

class DistanceReward(RewardFunction):
    """距离奖励 - 鼓励机械臂末端接近目标（线性）"""
    
    def __init__(self, weight: float = 0.01):
        """
        初始化距离奖励
        
        Args:
            weight: 权重，默认0.01（相当于原来的1.0/100）
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算距离奖励（线性）
        
        公式: reward = -distance
        距离越小，奖励越高（越接近0）
        
        Args:
            env_state: 环境状态，需要包含'distance'键
            
        Returns:
            float: 距离奖励（负值，距离越小越接近0）
        """
        distance = env_state['distance']  # 获取末端到目标的距离
        return -distance  # 返回负距离


class SquaredDistanceReward(RewardFunction):
    """平方距离奖励 - 更强烈地惩罚远距离，鼓励快速接近目标"""
    
    def __init__(self, weight: float = 0.0001):
        """
        初始化平方距离奖励
        
        Args:
            weight: 权重，默认0.0001（因为平方后数值会变大，所以权重要小）
        
        注意：
            - 平方奖励的数值范围比线性大得多
            - 距离100时: 线性=-100, 平方=-10000
            - 因此默认权重从0.01降到0.0001
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算平方距离奖励
        
        公式: reward = -distance²
        距离越小，奖励越高；远距离惩罚更重
        
        特点：
            - 远距离时惩罚非常重，鼓励快速接近
            - 近距离时惩罚相对较轻
            - 梯度随距离增大而增大
        
        示例：
            - 距离10: -10² = -100
            - 距离50: -50² = -2500
            - 距离100: -100² = -10000
        
        Args:
            env_state: 环境状态，需要包含'distance'键
            
        Returns:
            float: 平方距离奖励（负值，远距离惩罚更重）
        """
        distance = env_state['distance']  # 获取末端到目标的距离
        return -(distance ** 2)  # 返回负的距离平方


class SuccessReward(RewardFunction):
    """成功奖励 - 当机械臂到达目标时给予额外奖励"""
    
    def __init__(self, weight: float = 10.0, threshold: float = 10.0):
        """
        初始化成功奖励
        
        Args:
            weight: 权重（也是成功时的奖励值），默认10.0
            threshold: 成功判定阈值（像素），默认10.0
        """
        super().__init__(weight)  # 调用父类初始化
        self.threshold = threshold  # 保存成功阈值
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算成功奖励
        
        公式: reward = 1.0 if distance < threshold else 0
        只有到达目标时才给奖励
        
        Args:
            env_state: 环境状态，需要包含'distance'键
            
        Returns:
            float: 成功奖励（到达目标时为1.0，否则为0）
        """
        distance = env_state['distance']  # 获取距离
        return 1.0 if distance < self.threshold else 0.0  # 判断是否成功


class VelocityPenalty(RewardFunction):
    """速度惩罚 - 惩罚过快的运动，鼓励平滑控制"""
    
    def __init__(self, weight: float = 0.01):
        """
        初始化速度惩罚
        
        Args:
            weight: 权重，默认0.01（相当于原来的1.0/100）
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算速度惩罚
        
        公式: penalty = -||velocities||
        速度越大，惩罚越大（负值越大）
        
        Args:
            env_state: 环境状态，需要包含'joint_velocities'键
            
        Returns:
            float: 速度惩罚（负值）
        """
        velocities = env_state.get('joint_velocities', [0, 0])  # 获取关节速度，默认[0,0]
        velocity_magnitude = np.linalg.norm(velocities)  # 计算速度向量的模（大小）
        return -velocity_magnitude  # 返回负速度


class ActionPenalty(RewardFunction):
    """动作惩罚 - 惩罚大的动作，鼓励节能控制"""
    
    def __init__(self, weight: float = 0.1):
        """
        初始化动作惩罚
        
        Args:
            weight: 权重，默认0.1（相当于原来的1.0/10）
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算动作惩罚
        
        公式: penalty = -||action||
        动作幅度越大，惩罚越大
        
        Args:
            env_state: 环境状态，需要包含'action'键
            
        Returns:
            float: 动作惩罚（负值）
        """
        action = env_state.get('action', [0, 0])  # 获取动作，默认[0,0]
        action_magnitude = np.linalg.norm(action)  # 计算动作向量的模
        return -action_magnitude  # 返回负动作幅度


class AccelerationPenalty(RewardFunction):
    """加速度惩罚 - 惩罚剧烈的加速度变化，鼓励平滑运动"""
    
    def __init__(self, weight: float = 0.01):
        """
        初始化加速度惩罚
        
        Args:
            weight: 权重，默认0.01（相当于原来的1.0/100）
        """
        super().__init__(weight)  # 调用父类初始化
        self.prev_velocities = None  # 保存上一步的速度，用于计算加速度
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算加速度惩罚
        
        公式: penalty = -||acceleration||
        加速度 = 当前速度 - 上一步速度
        
        Args:
            env_state: 环境状态，需要包含'joint_velocities'键
            
        Returns:
            float: 加速度惩罚（负值）
        """
        velocities = env_state.get('joint_velocities', [0, 0])  # 获取当前速度
        
        if self.prev_velocities is None:  # 如果是第一步
            self.prev_velocities = velocities  # 保存当前速度
            return 0.0  # 第一步没有加速度，返回0
        
        # 计算加速度 = 当前速度 - 上一步速度
        acceleration = np.array(velocities) - np.array(self.prev_velocities)
        self.prev_velocities = velocities  # 更新上一步速度
        
        acceleration_magnitude = np.linalg.norm(acceleration)  # 计算加速度的模
        return -acceleration_magnitude  # 返回负加速度
    
    def reset(self):
        """重置状态（每个回合开始时调用）"""
        self.prev_velocities = None  # 清空上一步速度


class ProgressReward(RewardFunction):
    """进度奖励 - 奖励距离的改善，鼓励持续接近目标"""
    
    def __init__(self, weight: float = 0.01):
        """
        初始化进度奖励
        
        Args:
            weight: 权重，默认0.01（相当于原来的1.0/100）
        """
        super().__init__(weight)  # 调用父类初始化
        self.prev_distance = None  # 保存上一步的距离，用于计算进度
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算进度奖励
        
        公式: reward = prev_distance - distance
        进度 = 上一步距离 - 当前距离
        正值表示接近目标，负值表示远离目标
        
        Args:
            env_state: 环境状态，需要包含'distance'键
            
        Returns:
            float: 进度奖励（正值=接近，负值=远离）
        """
        distance = env_state['distance']  # 获取当前距离
        
        if self.prev_distance is None:  # 如果是第一步
            self.prev_distance = distance  # 保存当前距离
            return 0.0  # 第一步没有进度，返回0
        
        progress = self.prev_distance - distance  # 计算进度（正值=接近目标）
        self.prev_distance = distance  # 更新上一步距离
        
        return progress  # 返回进度
    
    def reset(self):
        """重置状态（每个回合开始时调用）"""
        self.prev_distance = None  # 清空上一步距离


class OrientationReward(RewardFunction):
    """方向奖励 - 奖励末端朝向目标，鼓励正确的姿态"""
    
    def __init__(self, weight: float = 1.0):
        """
        初始化方向奖励
        
        Args:
            weight: 权重，默认1.0
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算方向奖励
        
        公式: reward = dot(direction, orientation)
        计算末端朝向与目标方向的相似度
        
        Args:
            env_state: 环境状态，需要包含'end_effector_pos', 'target_pos', 'joint_angles'
            
        Returns:
            float: 方向奖励（-1到1，1表示完全对齐）
        """
        end_pos = env_state['end_effector_pos']  # 获取末端位置
        target_pos = env_state['target_pos']  # 获取目标位置
        
        # 计算末端到目标的方向向量
        direction = target_pos - end_pos
        direction_norm = np.linalg.norm(direction)  # 计算方向向量的模
        
        if direction_norm < 1e-6:  # 如果已经到达目标（距离极小）
            return 1.0  # 返回最大奖励
        
        direction = direction / direction_norm  # 归一化方向向量
        
        # 获取末端的朝向（简化：使用两个关节角度之和）
        joint_angles = env_state.get('joint_angles', [0, 0])  # 获取关节角度
        # 计算末端朝向向量（使用三角函数）
        end_orientation = np.array([np.cos(sum(joint_angles)), np.sin(sum(joint_angles))])
        
        # 计算方向相似度（点积，范围-1到1）
        alignment = np.dot(direction, end_orientation)
        return alignment  # 返回对齐度


class TimeEfficiencyReward(RewardFunction):
    """时间效率奖励 - 鼓励快速完成任务，避免拖延"""
    
    def __init__(self, weight: float = 1.0):
        """
        初始化时间效率奖励
        
        Args:
            weight: 权重，默认1.0（每步惩罚-1，通过weight调整）
        """
        super().__init__(weight)  # 调用父类初始化
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算时间效率奖励
        
        公式: reward = -1
        每步给予固定负奖励，鼓励快速完成
        
        Args:
            env_state: 环境状态
            
        Returns:
            float: 时间惩罚（-1）
        """
        # 每步给予固定的负奖励，鼓励快速完成任务
        return -1.0


class FailurePenalty(RewardFunction):
    """失败惩罚 - 当回合结束时未完成任务给予惩罚"""
    
    def __init__(self, weight: float = 10.0, threshold: float = 10.0):
        """
        初始化失败惩罚
        
        Args:
            weight: 权重（惩罚值），默认10.0
            threshold: 成功判定阈值（像素），默认10.0
        """
        super().__init__(weight)  # 调用父类初始化
        self.threshold = threshold  # 保存成功阈值
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算失败惩罚
        
        公式: penalty = -1.0 if (done and distance >= threshold) else 0
        只在回合结束且未成功时给予惩罚
        
        Args:
            env_state: 环境状态，需要包含'distance', 'done'键
            
        Returns:
            float: 失败惩罚（回合结束且失败时为-1.0，否则为0）
        """
        distance = env_state.get('distance', float('inf'))  # 获取距离
        done = env_state.get('done', False)  # 获取是否结束
        
        # 只在回合结束且未达到目标时给予惩罚
        if done and distance >= self.threshold:
            return -1.0  # 失败惩罚
        else:
            return 0.0  # 未结束或已成功，无惩罚


class ArrivalVelocityPenalty(RewardFunction):
    """到达速度惩罚 - 惩罚高速到达目标，鼓励轻柔触碰"""
    
    def __init__(self, weight: float = 200.0, threshold: float = 0.15):
        """
        初始化到达速度惩罚
        
        Args:
            weight: 权重，默认200.0
            threshold: 成功判定阈值（米），默认0.15
        """
        super().__init__(weight)
        self.threshold = threshold
        self.prev_distance = None
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算到达速度惩罚
        
        公式: penalty = -||end_effector_velocity|| (只在刚成功时)
        
        Args:
            env_state: 环境状态，需要包含'distance', 'end_effector_velocity'
            
        Returns:
            float: 到达速度惩罚（刚成功时为负值，否则为0）
        """
        distance = env_state.get('distance', float('inf'))
        
        # 检测是否刚刚成功（从失败变成功）
        just_succeeded = False
        if self.prev_distance is not None:
            if self.prev_distance >= self.threshold and distance < self.threshold:
                just_succeeded = True
        
        self.prev_distance = distance
        
        if just_succeeded:
            # 计算末端线速度
            end_velocity = env_state.get('end_effector_velocity', np.array([0, 0]))
            velocity_magnitude = np.linalg.norm(end_velocity)
            return -velocity_magnitude
        
        return 0.0
    
    def reset(self):
        """重置状态"""
        self.prev_distance = None


class PathLengthPenalty(RewardFunction):
    """路径长度惩罚 - 每步惩罚移动距离，鼓励少动、直达"""
    
    def __init__(self, weight: float = 1.0):
        """
        初始化路径长度惩罚
        
        Args:
            weight: 权重，默认1.0（每米移动惩罚1分）
        """
        super().__init__(weight)
        self.prev_pos = None
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算路径长度惩罚
        
        公式: penalty = -step_distance (每步)
        
        Args:
            env_state: 环境状态
            
        Returns:
            float: 路径长度惩罚（负值）
        """
        current_pos = env_state.get('end_effector_pos', np.array([0, 0]))
        
        if self.prev_pos is None:
            self.prev_pos = current_pos.copy()
            return 0.0
        
        # 计算这一步移动的距离
        step_distance = np.linalg.norm(current_pos - self.prev_pos)
        self.prev_pos = current_pos.copy()
        
        # 返回负的移动距离（惩罚）
        return -step_distance
    
    def reset(self):
        """重置状态"""
        self.prev_pos = None


class PathEfficiencyPenalty(RewardFunction):
    """路径效率惩罚 - 惩罚绕路，鼓励直线路径"""
    
    def __init__(self, weight: float = 1000.0, threshold: float = 0.15):
        """
        初始化路径效率惩罚
        
        Args:
            weight: 权重，默认1000.0
            threshold: 成功判定阈值（米），默认0.15
        """
        super().__init__(weight)
        self.threshold = threshold
        self.initial_distance = None
        self.total_path_length = 0.0
        self.prev_pos = None
        self.prev_distance = None
    
    def compute(self, env_state: Dict[str, Any]) -> float:
        """
        计算路径效率惩罚
        
        公式: penalty = -(1.0 - efficiency) (只在刚成功时)
        其中 efficiency = initial_distance / total_path_length
        
        Args:
            env_state: 环境状态
            
        Returns:
            float: 路径效率惩罚（刚成功时计算，否则为0）
        """
        distance = env_state.get('distance', float('inf'))
        current_pos = env_state.get('end_effector_pos', np.array([0, 0]))
        
        # 记录初始距离
        if self.initial_distance is None:
            self.initial_distance = distance
        
        # 累积路径长度
        if self.prev_pos is not None:
            step_distance = np.linalg.norm(current_pos - self.prev_pos)
            self.total_path_length += step_distance
        
        self.prev_pos = current_pos.copy()
        
        # 检测是否刚刚成功
        just_succeeded = False
        if self.prev_distance is not None:
            if self.prev_distance >= self.threshold and distance < self.threshold:
                just_succeeded = True
        
        self.prev_distance = distance
        
        if just_succeeded and self.total_path_length > 0:
            # 计算路径效率
            efficiency = self.initial_distance / self.total_path_length
            # efficiency = 1.0 表示完美直线
            # efficiency < 1.0 表示绕路
            penalty = -(1.0 - efficiency)
            return penalty
        
        return 0.0
    
    def reset(self):
        """重置状态"""
        self.initial_distance = None
        self.total_path_length = 0.0
        self.prev_pos = None
        self.prev_distance = None


# ============================================================================
# 奖励管理器
# ============================================================================

class RewardManager:
    """奖励管理器 - 组合和管理多个奖励函数，计算总奖励"""
    
    def __init__(self, reward_components: Dict[str, RewardFunction] = None):
        """
        初始化奖励管理器
        
        Args:
            reward_components: 奖励组件字典 {名称: 奖励函数对象}
                              例如: {'distance': DistanceReward(), 'success': SuccessReward()}
        """
        self.reward_components = reward_components or {}  # 保存奖励组件，如果为None则初始化为空字典
        # 为每个组件创建历史记录列表
        self.reward_history = {name: [] for name in self.reward_components.keys()}
        self.reward_history['total'] = []  # 添加总奖励的历史记录
    
    def add_component(self, name: str, reward_fn: RewardFunction):
        """
        添加奖励组件
        
        Args:
            name: 组件名称，例如'distance', 'success'
            reward_fn: 奖励函数对象
        """
        self.reward_components[name] = reward_fn  # 添加到组件字典
        self.reward_history[name] = []  # 为该组件创建历史记录列表
    
    def remove_component(self, name: str):
        """
        移除奖励组件
        
        Args:
            name: 要移除的组件名称
        """
        if name in self.reward_components:  # 如果组件存在
            del self.reward_components[name]  # 从组件字典中删除
            del self.reward_history[name]  # 从历史记录中删除
    
    def compute_reward(self, env_state: Dict[str, Any], log: bool = True) -> tuple:
        """
        计算总奖励（核心方法）
        
        遍历所有奖励组件，计算各自的奖励值，然后累加得到总奖励
        
        Args:
            env_state: 环境状态字典，包含距离、位置、速度等信息
            log: 是否记录各组件的奖励到历史记录，默认True
        
        Returns:
            tuple: (total_reward, reward_info)
                - total_reward: 总奖励值（所有组件的加权和）
                - reward_info: 各组件的奖励详情字典 {组件名: 奖励值}
        """
        total_reward = 0.0  # 初始化总奖励
        reward_info = {}  # 初始化奖励详情字典
        
        # 遍历所有奖励组件
        for name, reward_fn in self.reward_components.items():
            component_reward = reward_fn(env_state)  # 调用组件计算奖励（自动加权）
            total_reward += component_reward  # 累加到总奖励
            reward_info[name] = component_reward  # 记录该组件的奖励
            
            if log:  # 如果需要记录历史
                self.reward_history[name].append(component_reward)  # 添加到历史记录
        
        if log:  # 如果需要记录历史
            self.reward_history['total'].append(total_reward)  # 记录总奖励
        
        return total_reward, reward_info  # 返回总奖励和详情
    
    def reset(self):
        """
        重置有状态的奖励组件
        
        某些组件（如ProgressReward, AccelerationPenalty）需要保存上一步的状态
        每个回合开始时需要调用此方法清空状态
        """
        for reward_fn in self.reward_components.values():  # 遍历所有组件
            if hasattr(reward_fn, 'reset'):  # 如果组件有reset方法
                reward_fn.reset()  # 调用reset清空状态
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取奖励统计信息
        
        计算各组件历史奖励的统计指标（均值、标准差、最小值、最大值、总和）
        
        Returns:
            Dict: 统计信息字典
                {
                    '组件名': {
                        'mean': 均值,
                        'std': 标准差,
                        'min': 最小值,
                        'max': 最大值,
                        'sum': 总和
                    }
                }
        """
        stats = {}  # 初始化统计字典
        for name, history in self.reward_history.items():  # 遍历所有历史记录
            if len(history) > 0:  # 如果有记录
                stats[name] = {  # 计算统计指标
                    'mean': np.mean(history),  # 均值
                    'std': np.std(history),  # 标准差
                    'min': np.min(history),  # 最小值
                    'max': np.max(history),  # 最大值
                    'sum': np.sum(history),  # 总和
                }
        return stats  # 返回统计信息
    
    def clear_history(self):
        """
        清空历史记录
        
        清空所有组件的历史奖励记录
        """
        for name in self.reward_history.keys():  # 遍历所有历史记录
            self.reward_history[name] = []  # 清空列表


# ============================================================================
# 预设奖励配置
# ============================================================================

def create_default_reward_manager(config: Dict[str, Any] = None) -> RewardManager:
    """
    创建默认奖励管理器（基础模式）
    
    包含两个基础奖励组件：
    1. 距离奖励 - 鼓励接近目标（线性或平方）
    2. 成功奖励 - 到达目标时的额外奖励
    
    适用场景：快速训练、基础任务
    
    Args:
        config: 配置字典，包含各组件的参数
            - use_squared_distance: bool, 是否使用平方距离奖励（默认False）
            - distance_weight: float, 距离奖励权重
            - success_weight: float, 成功奖励权重
            - success_threshold: float, 成功判定阈值
        
    Returns:
        RewardManager: 配置好的奖励管理器
    """
    if config is None:  # 如果没有提供配置
        config = {}  # 使用空字典（将使用默认值）
    
    manager = RewardManager()  # 创建空的奖励管理器
    
    # 根据配置选择距离奖励类型
    use_squared = config.get('use_squared_distance', False)  # 是否使用平方距离
    
    if use_squared:
        # 使用平方距离奖励（权重默认更小）
        manager.add_component('distance', SquaredDistanceReward(
            weight=config.get('distance_weight', 0.0001)  # 平方奖励默认0.0001
        ))
    else:
        # 使用线性距离奖励
        manager.add_component('distance', DistanceReward(
            weight=config.get('distance_weight', 0.01)  # 线性奖励默认0.01
        ))
    
    # 添加成功奖励组件
    manager.add_component('success', SuccessReward(
        weight=config.get('success_weight', 10.0),  # 从配置获取权重，默认10.0
        threshold=config.get('success_threshold', 10.0)  # 从配置获取阈值，默认10.0
    ))
    
    return manager  # 返回配置好的管理器


def create_smooth_reward_manager(config: Dict[str, Any] = None) -> RewardManager:
    """
    创建平滑控制奖励管理器
    
    在默认奖励基础上添加：
    3. 速度惩罚 - 惩罚过快的运动
    4. 动作惩罚 - 惩罚大的动作
    
    适用场景：需要平滑、稳定控制的任务（如工业机器人）
    
    Args:
        config: 配置字典
            - use_squared_distance: bool, 是否使用平方距离奖励
            - 其他参数同create_default_reward_manager
        
    Returns:
        RewardManager: 配置好的奖励管理器
    """
    if config is None:  # 如果没有提供配置
        config = {}  # 使用空字典
    
    manager = create_default_reward_manager(config)  # 先创建默认管理器（包含距离和成功奖励）
    
    # 添加速度惩罚组件
    manager.add_component('velocity_penalty', VelocityPenalty(
        weight=config.get('velocity_penalty_weight', 0.01)  # 权重，默认0.01
    ))
    
    # 添加动作惩罚组件
    manager.add_component('action_penalty', ActionPenalty(
        weight=config.get('action_penalty_weight', 0.1)  # 权重，默认0.1
    ))
    
    # 添加到达速度惩罚组件（新增）
    manager.add_component('arrival_velocity_penalty', ArrivalVelocityPenalty(
        weight=config.get('arrival_velocity_penalty_weight', 200.0),
        threshold=config.get('success_threshold', 0.15)
    ))
    
    # 添加路径效率惩罚组件（新增）
    manager.add_component('path_efficiency_penalty', PathEfficiencyPenalty(
        weight=config.get('path_efficiency_penalty_weight', 1000.0),
        threshold=config.get('success_threshold', 0.15)
    ))
    
    # 添加进度奖励组件（引导选择最短路径）
    if config.get('progress_weight', 0.0) > 0:
        manager.add_component('progress', ProgressReward(
            weight=config.get('progress_weight', 0.0)
        ))
    
    # 添加路径长度惩罚组件（持续惩罚移动，鼓励少动）
    if config.get('path_length_penalty_weight', 0.0) > 0:
        manager.add_component('path_length_penalty', PathLengthPenalty(
            weight=config.get('path_length_penalty_weight', 0.0)
        ))
    
    return manager  # 返回配置好的管理器


def create_efficient_reward_manager(config: Dict[str, Any] = None) -> RewardManager:
    """
    创建高效控制奖励管理器
    
    在默认奖励基础上添加：
    3. 进度奖励 - 奖励距离的改善
    4. 时间效率奖励 - 鼓励快速完成
    
    适用场景：需要快速响应的任务（如竞速、紧急响应）
    
    Args:
        config: 配置字典
            - use_squared_distance: bool, 是否使用平方距离奖励
            - 其他参数同create_default_reward_manager
        
    Returns:
        RewardManager: 配置好的奖励管理器
    """
    if config is None:  # 如果没有提供配置
        config = {}  # 使用空字典
    
    manager = create_default_reward_manager(config)  # 先创建默认管理器
    
    # 添加进度奖励组件
    manager.add_component('progress', ProgressReward(
        weight=config.get('progress_weight', 0.01)  # 权重，默认0.01
    ))
    
    # 添加时间效率奖励组件
    manager.add_component('time_efficiency', TimeEfficiencyReward(
        weight=config.get('time_efficiency_weight', 0.002)  # 权重，默认0.002 (相当于1/500)
    ))
    
    return manager  # 返回配置好的管理器


def create_advanced_reward_manager(config: Dict[str, Any] = None) -> RewardManager:
    """
    创建高级奖励管理器（包含所有主要组件）
    
    包含6个奖励组件：
    1. 距离奖励 - 鼓励接近目标（线性或平方）
    2. 成功奖励 - 到达目标时的额外奖励
    3. 失败惩罚 - 回合结束未完成时的惩罚
    4. 速度惩罚 - 惩罚过快的运动
    5. 动作惩罚 - 惩罚大的动作
    6. 进度奖励 - 奖励距离的改善
    
    适用场景：复杂任务，需要平衡多个目标
    
    Args:
        config: 配置字典
            - use_squared_distance: bool, 是否使用平方距离奖励
            - 其他参数同各组件
        
    Returns:
        RewardManager: 配置好的奖励管理器
    """
    if config is None:  # 如果没有提供配置
        config = {}  # 使用空字典
    
    manager = RewardManager()  # 创建空的奖励管理器
    
    # === 基础奖励 ===
    # 根据配置选择距离奖励类型
    use_squared = config.get('use_squared_distance', False)
    
    if use_squared:
        # 使用平方距离奖励
        manager.add_component('distance', SquaredDistanceReward(
            weight=config.get('distance_weight', 0.0001)  # 平方奖励默认0.0001
        ))
    else:
        # 使用线性距离奖励
        manager.add_component('distance', DistanceReward(
            weight=config.get('distance_weight', 0.01)  # 线性奖励默认0.01
        ))
    
    # 添加成功奖励组件
    manager.add_component('success', SuccessReward(
        weight=config.get('success_weight', 10.0),  # 权重，默认10.0
        threshold=config.get('success_threshold', 10.0)  # 阈值，默认10.0
    ))
    
    # 添加失败惩罚组件
    manager.add_component('failure_penalty', FailurePenalty(
        weight=config.get('failure_penalty_weight', 10.0),  # 权重，默认10.0
        threshold=config.get('success_threshold', 10.0)  # 使用相同的成功阈值
    ))
    
    # === 平滑性奖励 ===
    # 添加速度惩罚组件
    manager.add_component('velocity_penalty', VelocityPenalty(
        weight=config.get('velocity_penalty_weight', 0.01)  # 权重，默认0.01
    ))
    
    # 添加动作惩罚组件
    manager.add_component('action_penalty', ActionPenalty(
        weight=config.get('action_penalty_weight', 0.1)  # 权重，默认0.1
    ))
    
    # === 进度奖励 ===
    # 添加进度奖励组件
    manager.add_component('progress', ProgressReward(
        weight=config.get('progress_weight', 0.0)  # 权重，默认0.0（已关闭）
    ))
    
    return manager  # 返回配置好的管理器


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：创建和使用奖励管理器
    
    # 1. 准备配置字典
    config = {
        'distance_reward_scale': 100.0,  # 距离奖励缩放因子
        'success_reward': 10.0,  # 成功奖励值
        'success_threshold': 10.0,  # 成功判定阈值
    }
    
    # 2. 创建默认奖励管理器
    manager = create_default_reward_manager(config)
    
    # 3. 模拟环境状态（实际使用时由环境提供）
    env_state = {
        'distance': 50.0,  # 末端到目标的距离
        'end_effector_pos': np.array([100, 100]),  # 末端位置
        'target_pos': np.array([150, 150]),  # 目标位置
        'joint_velocities': [0.5, 0.3],  # 关节速度
        'action': [0.2, 0.1],  # 执行的动作
    }
    
    # 4. 计算奖励
    total_reward, reward_info = manager.compute_reward(env_state)
    
    # 5. 打印结果
    print(f"Total Reward: {total_reward:.4f}")  # 打印总奖励
    print("Reward Components:")  # 打印各组件的奖励
    for name, value in reward_info.items():  # 遍历所有组件
        print(f"  {name}: {value:.4f}")  # 打印组件名和奖励值
