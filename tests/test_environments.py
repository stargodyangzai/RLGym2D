"""
环境测试
"""
import pytest
import numpy as np
from envs import make_env
from configs import TASK_CONFIGS


class TestEnvironments:
    """测试所有环境"""
    
    @pytest.mark.parametrize("task", ["cartpole", "arm", "walker"])
    def test_env_creation(self, task):
        """测试环境创建"""
        config = TASK_CONFIGS[task]['env_config']
        env = make_env(task, render_mode=None, config=config)
        assert env is not None
        
    @pytest.mark.parametrize("task", ["cartpole", "arm", "walker"])
    def test_env_reset(self, task):
        """测试环境重置"""
        config = TASK_CONFIGS[task]['env_config']
        env = make_env(task, render_mode=None, config=config)
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
    @pytest.mark.parametrize("task", ["cartpole", "arm", "walker"])
    def test_env_step(self, task):
        """测试环境步进"""
        config = TASK_CONFIGS[task]['env_config']
        env = make_env(task, render_mode=None, config=config)
        obs, info = env.reset()
        
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])