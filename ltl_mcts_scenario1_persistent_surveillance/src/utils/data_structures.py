"""
Core data structure definitions
Contains agent state, target state, world state, observation data and other class definitions
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

@dataclass
class AgentState:
    """Agent state"""
    position: np.ndarray  # [x, y] position
    velocity: np.ndarray  # [vx, vy] velocity
    heading: float        # current heading angle (radians)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

@dataclass
class TargetState:
    """True state of a single target"""
    id: str                    # unique identifier
    is_alive: bool            # whether alive
    position: np.ndarray      # [x, y] position
    velocity: np.ndarray      # [vx, vy] velocity
    birth_time: int           # birth time step
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
    
    def get_full_state(self) -> np.ndarray:
        """Return full state vector [x, y, vx, vy]"""
        return np.concatenate([self.position, self.velocity])

class Measurement:
    """Single measurement point"""
    def __init__(self, value: np.ndarray, is_clutter: bool = False, 
                 source_target_id: Optional[str] = None):
        self.value = np.asarray(value, dtype=float)  # [z_x, z_y]
        self.is_clutter = is_clutter                 # whether it's clutter
        self.source_target_id = source_target_id     # source target ID (if not clutter)

class Observation:
    """All observation data within one time step"""
    def __init__(self, measurements: List[Measurement] = None):
        self.measurements = measurements or []
        self.timestamp = None
    
    def add_measurement(self, measurement: Measurement):
        """Add a measurement point"""
        self.measurements.append(measurement)
    
    def get_measurement_array(self) -> np.ndarray:
        """Return numpy array of all measurement points, shape (n_measurements, 2)"""
        if not self.measurements:
            return np.empty((0, 2))
        return np.array([m.value for m in self.measurements])
    
    def __len__(self):
        return len(self.measurements)

class WorldState:
    """Complete world state"""
    def __init__(self, time_step: int = 0):
        self.time_step = time_step
        self.agent_state: Optional[AgentState] = None
        self.targets: List[TargetState] = []
        self.observation_history: List[Observation] = []
    
    def add_target(self, target: TargetState):
        """Add a new target"""
        self.targets.append(target)
    
    def get_alive_targets(self) -> List[TargetState]:
        """Get all alive targets"""
        return [t for t in self.targets if t.is_alive]
    
    def get_target_by_id(self, target_id: str) -> Optional[TargetState]:
        """Get target by ID"""
        for target in self.targets:
            if target.id == target_id:
                return target
        return None

# ========== PMBM filter related data structures ==========

class BernoulliComponent:
    """Bernoulli component - represents a hypothesis of a tracked target"""
    def __init__(self, existence_prob: float, state_mean: np.ndarray, 
                 state_cov: np.ndarray, label: Optional[str] = None):
        self.existence_prob = existence_prob  # existence probability r
        self.state_mean = np.asarray(state_mean, dtype=float)      # state mean μ
        self.state_cov = np.asarray(state_cov, dtype=float)        # state covariance P
        self.label = label or str(uuid.uuid4())  # label
        
    def get_position_mean(self) -> np.ndarray:
        """Get position estimate mean [x, y]"""
        return self.state_mean[:2]
    
    def get_position_cov(self) -> np.ndarray:
        """Get position estimate covariance 2x2 matrix"""
        return self.state_cov[:2, :2]
    
    def get_covariance_trace(self) -> float:
        """Get the trace of the position covariance matrix"""
        return np.trace(self.get_position_cov())
    
    def is_confirmed(self, threshold: float) -> bool:
        """Check if the target is confirmed (covariance trace less than threshold)"""
        return self.get_covariance_trace() < threshold

class PoissonComponent:
    """Poisson component - represents the distribution of undetected targets"""
    def __init__(self, intensity_function):
        self.intensity_function = intensity_function  # Intensity function λ(x)
    
    def get_expected_number_in_region(self, region) -> float:
        """Compute the expected number of targets in the given region
        Simplified: if the intensity function is uniform λ(x)=λ0, then E[#|R]≈λ0 * Area(R)
        """
        try:
            rate = float(self.intensity_function(None)) if callable(self.intensity_function) else 0.0
        except Exception:
            rate = 0.0
        try:
            area = float((region[1] - region[0]) * (region[3] - region[2]))
        except Exception:
            area = 0.0
        return max(0.0, rate * area)

class GlobalHypothesis:
    """Global hypothesis - a possible data association hypothesis"""
    def __init__(self, weight: float, bernoulli_components: List[BernoulliComponent]):
        self.weight = weight                                    # Hypothesis weight
        self.bernoulli_components = bernoulli_components        # Bernoulli component list
    
    def get_map_estimate(self) -> List[BernoulliComponent]:
        """Get the list of targets with the maximum posterior estimate"""
        # Return Bernoulli components with existence probability greater than 0.3 (balance quality and quantity)
        return [bc for bc in self.bernoulli_components if bc.existence_prob > 0.3]

class PMBMBelief:
    """PMBM belief state"""
    def __init__(self):
        self.poisson_component: Optional[PoissonComponent] = None
        self.global_hypotheses: List[GlobalHypothesis] = []
        self.timestamp: int = 0
    
    def get_map_hypothesis(self) -> Optional[GlobalHypothesis]:
        """Get the global hypothesis with the maximum posterior probability"""
        if not self.global_hypotheses:
            return None
        return max(self.global_hypotheses, key=lambda h: h.weight)
    
    def get_map_targets(self) -> List[BernoulliComponent]:
        """Get the list of targets with the MAP estimate"""
        map_hyp = self.get_map_hypothesis()
        if map_hyp is None:
            return []
        return map_hyp.get_map_estimate()
    
    def get_top_k_hypotheses(self, k: int = 2) -> List[GlobalHypothesis]:
        """Get the top k hypotheses with the highest weight"""
        sorted_hyps = sorted(self.global_hypotheses, key=lambda h: h.weight, reverse=True)
        return sorted_hyps[:k]
    
    def get_weight_gap(self) -> float:
        """Get the weight difference between the top two hypotheses"""
        top_hyps = self.get_top_k_hypotheses(2)
        if len(top_hyps) < 2:
            return 1.0  # When there is only one hypothesis, the weight difference is 1
        return top_hyps[0].weight - top_hyps[1].weight

# ========== LTL related data structures ==========

@dataclass
class AtomicProposition:
    """Atomic proposition"""
    name: str                    # Proposition name, such as "ap_confirm_any"
    probability: float = 0.0     # Satisfied probability
    is_satisfied: bool = False   # Whether satisfied (after thresholding)

class LTLState:
    """LTL automaton state"""
    def __init__(self, automaton_state: int, memory_state: Dict[str, bool]):
        self.automaton_state = automaton_state      # LDBA state
        self.memory_state = memory_state.copy()     # Delayed memory state
    
    def __eq__(self, other):
        if not isinstance(other, LTLState):
            return False
        return (self.automaton_state == other.automaton_state and 
                self.memory_state == other.memory_state)
    
    def __hash__(self):
        memory_tuple = tuple(sorted(self.memory_state.items()))
        return hash((self.automaton_state, memory_tuple))

    # ========== Planner related data structures ==========

class MCTSNode:
    """MCTS tree node"""
    def __init__(self, state_summary: np.ndarray, ltl_state: LTLState, 
                 parent=None, action=None):
        self.state_summary = state_summary      # LTL belief summary b^φ
        self.ltl_state = ltl_state             # LTL state (q, m)
        self.parent = parent                   # Parent node
        self.action = action                   # Action to reach this node
        
        self.children: Dict[int, 'MCTSNode'] = {}  # Child node dictionary {action: child_node}
        self.visits = 0                        # Visits
        self.total_reward = 0.0               # Cumulative reward
        self.is_terminal = False              # Whether terminal node
        
    def is_fully_expanded(self, num_actions: int) -> bool:
        """Check if fully expanded"""
        return len(self.children) == num_actions
    
    def get_ucb1_score(self, exploration_constant: float) -> float:
        """Compute UCB1 score"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_reward / self.visits
        if self.parent is None or self.parent.visits == 0:
            exploration = 0
        else:
            exploration = exploration_constant * np.sqrt(
                np.log(self.parent.visits) / self.visits
            )
        
        return exploitation + exploration
    
    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.total_reward += reward
    
    def get_best_child(self) -> Tuple[int, 'MCTSNode']:
        """Get the child node with the most visits"""
        if not self.children:
            return None, None
        
        best_action, best_child = max(
            self.children.items(), 
            key=lambda item: item[1].visits
        )
        return best_action, best_child

# ========== Useful functions ==========

def create_uniform_poisson_component(intensity_rate: float = 0.05) -> PoissonComponent:
    """Create uniform Poisson component"""
    def uniform_intensity(x):
        return intensity_rate
    
    return PoissonComponent(uniform_intensity)

def create_initial_pmbm_belief() -> PMBMBelief:
    """Create initial PMBM belief"""
    belief = PMBMBelief()
    # Use PPP density in config, avoid region prior existence probability too strong
    try:
        # Get PPP_DENSITY from config
        from configs.config import PPP_DENSITY
        ppp_rate = float(PPP_DENSITY)
    except ImportError:
        ppp_rate = 1e-6  # Conservative default
    belief.poisson_component = create_uniform_poisson_component(ppp_rate)
    belief.global_hypotheses = [GlobalHypothesis(1.0, [])]  # Initial assumption: no target
    return belief
