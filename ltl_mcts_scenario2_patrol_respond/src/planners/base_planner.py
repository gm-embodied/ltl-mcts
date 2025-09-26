"""
基础规划器抽象类和通用功能
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Tuple

class BasePlanner(ABC):
    """规划器基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.planning_time_history = []
        self.action_history = []
    
    @abstractmethod
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        根据当前状态规划下一步动作
        
        Args:
            world_state: 世界状态
            pmbm_belief: PMBM信念状态
            ltl_state: LTL自动机状态
            
        Returns:
            int: 动作索引 (0-7对应8个控制角度)
        """
        pass
    
    def reset(self):
        """重置规划器状态"""
        self.planning_time_history = []
        self.action_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取规划器统计信息"""
        return {
            'name': self.name,
            'total_planning_calls': len(self.planning_time_history),
            'average_planning_time': np.mean(self.planning_time_history) if self.planning_time_history else 0,
            'action_distribution': self._compute_action_distribution()
        }
    
    def _compute_action_distribution(self) -> Dict[int, int]:
        """计算动作分布"""
        distribution = {i: 0 for i in range(8)}
        for action in self.action_history:
            if 0 <= action <= 7:
                distribution[action] += 1
        return distribution
