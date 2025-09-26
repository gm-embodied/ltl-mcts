"""
Base planner abstract class and common functionality
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, List, Tuple

class BasePlanner(ABC):
    """Base class for planners"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.planning_time_history = []
        self.action_history = []
    
    @abstractmethod
    def plan_action(self, world_state, pmbm_belief, ltl_state) -> int:
        """
        Plan next action based on current state
        
        Args:
            world_state: world state
            pmbm_belief: PMBM belief state
            ltl_state: LTL automaton state
            
        Returns:
            int: action index (0-7 corresponding to 8 control angles)
        """
        pass
    
    def reset(self):
        """Reset planner state"""
        self.planning_time_history = []
        self.action_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            'name': self.name,
            'total_planning_calls': len(self.planning_time_history),
            'average_planning_time': np.mean(self.planning_time_history) if self.planning_time_history else 0,
            'action_distribution': self._compute_action_distribution()
        }
    
    def _compute_action_distribution(self) -> Dict[int, int]:
        """Compute action distribution"""
        distribution = {i: 0 for i in range(8)}
        for action in self.action_history:
            if 0 <= action <= 7:
                distribution[action] += 1
        return distribution
