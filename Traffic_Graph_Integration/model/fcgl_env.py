"""
FCGL Supply Chain Environment
Reinforcement learning environment for supply chain routing optimization
"""

from typing import Tuple, List, Dict, Any, Optional
from graph_env import GraphEnvironment


class FCGLSupplyChainEnv:
    """
    Supply Chain Environment for Flow-based Constrained Graph Learning (FCGL)
    
    State: (current_node, remaining_volume)
    Actions: List of outgoing edges from current node
    """
    
    def __init__(self, graph: GraphEnvironment, default_volume: float = 1000.0):
        """
        Initialize FCGL Supply Chain Environment
        
        Args:
            graph: GraphEnvironment instance with network topology
            default_volume: Default shipment volume in kg (default: 1000)
        """
        self.graph = graph
        self.default_volume = default_volume
        self.current_state = None
        
        # Statistics
        self.episode_count = 0
        self.step_count = 0
    
    def reset(self, start_node: str, volume: Optional[float] = None) -> Tuple[str, float]:
        """
        Reset environment to initial state
        
        Args:
            start_node: Starting node ID (typically a source node)
            volume: Shipment volume in kg (uses default_volume if None)
            
        Returns:
            Initial state tuple (node, remaining_volume)
        """
        if volume is None:
            volume = self.default_volume
        
        # Validate start node exists
        if start_node not in self.graph.get_all_nodes():
            raise ValueError(f"Start node '{start_node}' not found in graph")
        
        self.current_state = (start_node, volume)
        self.step_count = 0
        self.episode_count += 1
        
        return self.current_state
    
    def get_state(self) -> Tuple[str, float]:
        """
        Get current state
        
        Returns:
            Current state tuple (node, remaining_volume)
        """
        return self.current_state
    
    def actions(self, state: Tuple[str, float]) -> List[Dict[str, Any]]:
        """
        Get available actions (outgoing edges) from current state
        
        Args:
            state: Current state (node, remaining_volume)
            
        Returns:
            List of available actions (edges with their properties)
        """
        current_node, remaining_volume = state
        
        # Get all outgoing edges
        outgoing_edges = self.graph.get_outgoing(current_node)
        
        # Filter by capacity constraint if edge has capacity info
        feasible_actions = []
        for edge in outgoing_edges:
            # Check if edge can handle the remaining volume
            if 'capacity' in edge:
                if edge['capacity'] >= remaining_volume:
                    feasible_actions.append(edge)
            else:
                # If no capacity info, assume edge is feasible
                feasible_actions.append(edge)
        
        return feasible_actions
    
    def step(self, state: Tuple[str, float], action: Dict[str, Any]) -> Tuple[Tuple[str, float], float, bool]:
        """
        Take a step in the environment
        
        Args:
            state: Current state (node, remaining_volume)
            action: Selected action (edge dictionary)
            
        Returns:
            Tuple of (next_state, edge_cost, is_terminal)
            - next_state: (next_node, remaining_volume)
            - edge_cost: Cost of traversing the edge
            - is_terminal: Whether next state is terminal
        """
        current_node, remaining_volume = state
        
        # Extract next node from action
        next_node = action['to']
        
        # Volume remains the same (single-mode, no splitting)
        next_remaining_volume = remaining_volume
        
        # Get edge cost
        edge_cost = action['cost']
        
        # Create next state
        next_state = (next_node, next_remaining_volume)
        
        # Check if next state is terminal
        terminal = self.is_terminal(next_state)
        
        # Update internal state
        self.current_state = next_state
        self.step_count += 1
        
        return next_state, edge_cost, terminal
    
    def is_terminal(self, state: Tuple[str, float]) -> bool:
        """
        Check if state is terminal (reached a sink node)
        
        Args:
            state: State tuple (node, remaining_volume)
            
        Returns:
            True if node is a terminal (sink) node
        """
        node, _ = state
        return self.graph.is_terminal(node)
    
    def trajectory_to_reward(self, trajectory: List[Tuple[Tuple[str, float], Dict[str, Any]]]) -> Dict[str, float]:
        """
        Calculate reward metrics from a complete trajectory
        
        Args:
            trajectory: List of (state, action) tuples representing the path taken
            
        Returns:
            Dictionary with reward components:
                - total_cost: Sum of edge costs
                - total_co2: Sum of CO2 emissions
                - total_distance: Sum of distances
                - total_time: Sum of travel times
                - num_steps: Number of steps in trajectory
        """
        total_cost = 0.0
        total_co2 = 0.0
        total_distance = 0.0
        total_time = 0.0
        num_steps = len(trajectory)
        
        for state, action in trajectory:
            # Accumulate costs
            total_cost += action.get('cost', 0.0)
            total_co2 += action.get('co2', 0.0)
            total_distance += action.get('dist', 0.0)
            total_time += action.get('time', 0.0)
        
        return {
            'total_cost': total_cost,
            'total_co2': total_co2,
            'total_distance': total_distance,
            'total_time': total_time,
            'num_steps': num_steps
        }
    
    def get_reward(self, trajectory: List[Tuple[Tuple[str, float], Dict[str, Any]]], 
                   objective: str = 'cost') -> float:
        """
        Calculate scalar reward from trajectory based on objective
        
        Args:
            trajectory: List of (state, action) tuples
            objective: Optimization objective ('cost', 'co2', 'distance', 'time')
            
        Returns:
            Negative total (for minimization as RL typically maximizes reward)
        """
        metrics = self.trajectory_to_reward(trajectory)
        
        objective_map = {
            'cost': 'total_cost',
            'co2': 'total_co2',
            'distance': 'total_distance',
            'time': 'total_time'
        }
        
        metric_key = objective_map.get(objective, 'total_cost')
        
        # Return negative because RL maximizes reward, but we want to minimize cost
        return -metrics[metric_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics
        
        Returns:
            Dictionary with environment stats
        """
        return {
            'episode_count': self.episode_count,
            'current_step': self.step_count,
            'current_state': self.current_state,
            'num_nodes': len(self.graph.get_all_nodes()),
            'num_sources': len(self.graph.get_sources()),
            'num_terminals': len(self.graph.get_terminals())
        }
    
    def __repr__(self):
        return (f"FCGLSupplyChainEnv(default_volume={self.default_volume}, "
                f"episodes={self.episode_count}, "
                f"current_state={self.current_state})")


# Helper function to create environment
def create_fcgl_env(graph_csv_path: str, default_volume: float = 1000.0) -> FCGLSupplyChainEnv:
    """
    Create FCGL environment from graph CSV
    
    Args:
        graph_csv_path: Path to adjacency list CSV (supports multi-modal format)
        default_volume: Default shipment volume
        
    Returns:
        FCGLSupplyChainEnv instance
    """
    from graph_env import load_graph
    
    graph = load_graph(graph_csv_path)
    env = FCGLSupplyChainEnv(graph, default_volume=default_volume)
    
    return env


if __name__ == "__main__":
    # Test the FCGL environment
    print("Testing FCGLSupplyChainEnv with Multi-Modal Network...")
    print("="*70)
    
    # Create environment
    env = create_fcgl_env('graph_config/adjacency_list_multimodal.csv', default_volume=1000.0)
    
    print(f"\n{env}")
    print(f"\nEnvironment Statistics:")
    for key, value in env.get_statistics().items():
        print(f"  {key}: {value}")
    
    # Test reset
    print("\n" + "="*70)
    print("TEST: Reset Environment")
    print("="*70)
    sources = env.graph.get_sources()
    start_node = sources[0]
    initial_state = env.reset(start_node, volume=1200.0)
    print(f"Reset to: {initial_state}")
    print(f"Start node: {start_node}")
    print(f"Volume: {initial_state[1]} kg")
    
    # Test actions
    print("\n" + "="*70)
    print("TEST: Available Actions")
    print("="*70)
    actions = env.actions(initial_state)
    print(f"Available actions from {start_node}: {len(actions)}")
    for i, action in enumerate(actions[:3]):  # Show first 3
        print(f"  Action {i+1}: → {action['to']} "
              f"(cost={action['cost']:.2f}, co2={action['co2']:.2f}, "
              f"dist={action['dist']:.2f})")
    
    # Test step
    print("\n" + "="*70)
    print("TEST: Take a Step")
    print("="*70)
    if actions:
        selected_action = actions[0]
        next_state, cost, terminal = env.step(initial_state, selected_action)
        print(f"Action: {initial_state[0]} → {selected_action['to']}")
        print(f"Cost: {cost:.2f}")
        print(f"Next state: {next_state}")
        print(f"Is terminal: {terminal}")
        
        # Continue until terminal
        print("\n" + "="*70)
        print("TEST: Complete Trajectory")
        print("="*70)
        
        trajectory = [(initial_state, selected_action)]
        current = next_state
        max_steps = 20
        
        while not env.is_terminal(current) and len(trajectory) < max_steps:
            available_actions = env.actions(current)
            if not available_actions:
                print("No available actions - dead end!")
                break
            
            # Take first available action (greedy)
            action = available_actions[0]
            next_state, cost, terminal = env.step(current, action)
            trajectory.append((current, action))
            
            print(f"Step {len(trajectory)}: {current[0]} → {action['to']} "
                  f"(cost={cost:.2f})")
            
            current = next_state
        
        # Calculate rewards
        print("\n" + "="*70)
        print("TEST: Trajectory Rewards")
        print("="*70)
        rewards = env.trajectory_to_reward(trajectory)
        print("Trajectory metrics:")
        for key, value in rewards.items():
            print(f"  {key}: {value:.2f}")
        
        print("\nObjective-based rewards (negative for minimization):")
        for obj in ['cost', 'co2', 'distance', 'time']:
            reward = env.get_reward(trajectory, objective=obj)
            print(f"  {obj}: {reward:.2f}")
        
        print(f"\nFinal state: {current}")
        print(f"Reached terminal: {env.is_terminal(current)}")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
