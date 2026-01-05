"""
Reward Functions for FCGL Supply Chain Optimization
Multi-objective reward computation with cost and CO2 trade-offs
"""

import math
from typing import List, Dict, Any, Tuple


# ==================== CONSTANTS ====================
# Scaling factors to reduce large magnitudes
COST_SCALE = 1000.0       # cost / 1000
CO2_SCALE = 10000.0       # co2 / 10000
TIME_SCALE = 10.0         # time / 10

# Weights applied AFTER scaling
COST_WEIGHT = 0.05
CO2_WEIGHT = 0.01
TIME_WEIGHT = 0.05


# ==================== REWARD COMPUTATION ====================

def compute_reward(path_edges: List[Dict[str, Any]]) -> float:
    """
    Compute reward for a given path based on cost and CO2 emissions
    
    Uses exponential decay with weighted sum of cost and CO2:
    reward = exp(-(COST_WEIGHT * total_cost + CO2_WEIGHT * total_co2))
    
    Args:
        path_edges: List of edge dictionaries, each containing 'cost', 'co2', 'dist'
        
    Returns:
        Reward value (higher is better, range: 0 to 1)
    """
    if not path_edges:
        return 0.0
    
    # Compute total cost and CO2
    #total_cost = sum(edge.get('cost', 0.0) for edge in path_edges)
    #total_co2 = sum(edge.get('co2', 0.0) for edge in path_edges)
    
    # Compute weighted penalty
    #penalty = COST_WEIGHT * total_cost + CO2_WEIGHT * total_co2

    total_cost = sum(edge.get('cost', 0.0) for edge in path_edges) / COST_SCALE
    total_co2  = sum(edge.get('co2', 0.0) for edge in path_edges)  / CO2_SCALE
    total_time = sum(edge.get('time', 0.0) for edge in path_edges) / TIME_SCALE

    penalty = (
        COST_WEIGHT * total_cost +
        CO2_WEIGHT  * total_co2 +
        TIME_WEIGHT * total_time
    )


    
    # Exponential reward (decreases as cost/co2 increase)
    reward = math.exp(-penalty)
    
    return reward


def compute_reward_components(path_edges: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute detailed reward components for analysis
    
    Args:
        path_edges: List of edge dictionaries
        
    Returns:
        Dictionary with reward breakdown:
            - total_cost: Sum of edge costs
            - total_co2: Sum of CO2 emissions
            - total_distance: Sum of distances
            - total_time: Sum of travel times (if available)
            - penalty: Weighted penalty value
            - reward: Final reward value
    """
    if not path_edges:
        return {
            'total_cost': 0.0,
            'total_co2': 0.0,
            'total_distance': 0.0,
            'total_time': 0.0,
            'penalty': 0.0,
            'reward': 0.0
        }
    
    # Compute totals
    #total_cost = sum(edge.get('cost', 0.0) for edge in path_edges)
    #total_co2 = sum(edge.get('co2', 0.0) for edge in path_edges)
    total_distance = sum(edge.get('dist', 0.0) for edge in path_edges)
    #total_time = sum(edge.get('time', 0.0) for edge in path_edges)
    
    # Compute penalty and reward
    #penalty = COST_WEIGHT * total_cost + CO2_WEIGHT * total_co2

    total_cost = sum(edge.get('cost', 0.0) for edge in path_edges) / COST_SCALE
    total_co2  = sum(edge.get('co2', 0.0) for edge in path_edges)  / CO2_SCALE
    total_time = sum(edge.get('time', 0.0) for edge in path_edges) / TIME_SCALE

    penalty = (
        COST_WEIGHT * total_cost +
        CO2_WEIGHT  * total_co2 +
        TIME_WEIGHT * total_time
    )


    reward = math.exp(-penalty)
    
    return {
        'total_cost': total_cost,
        'total_co2': total_co2,
        'total_distance': total_distance,
        'total_time': total_time,
        'penalty': penalty,
        'reward': reward
    }


def compute_multi_objective_reward(path_edges: List[Dict[str, Any]], 
                                   cost_weight: float = COST_WEIGHT,
                                   co2_weight: float = CO2_WEIGHT,
                                   time_weight: float = 0.0) -> float:
    """
    Compute reward with customizable weights for different objectives
    
    Args:
        path_edges: List of edge dictionaries
        cost_weight: Weight for cost component
        co2_weight: Weight for CO2 component
        time_weight: Weight for time component (optional)
        
    Returns:
        Reward value
    """
    if not path_edges:
        return 0.0
    
    #total_cost = sum(edge.get('cost', 0.0) for edge in path_edges)
    #total_co2 = sum(edge.get('co2', 0.0) for edge in path_edges)
    #total_time = sum(edge.get('time', 0.0) for edge in path_edges)

    total_cost = sum(edge.get('cost', 0.0) for edge in path_edges) / COST_SCALE
    total_co2  = sum(edge.get('co2', 0.0) for edge in path_edges)  / CO2_SCALE
    total_time = sum(edge.get('time', 0.0) for edge in path_edges) / TIME_SCALE

    
    # Weighted penalty
    penalty = (cost_weight * total_cost + 
               co2_weight * total_co2 + 
               time_weight * total_time)
    
    reward = math.exp(-penalty)
    
    return reward


# ==================== PATH ANALYSIS ====================

def analyze_path(path_edges: List[Dict[str, Any]], 
                 path_nodes: List[str] = None) -> Dict[str, Any]:
    """
    Comprehensive path analysis with all metrics
    
    Args:
        path_edges: List of edge dictionaries
        path_nodes: Optional list of node IDs in the path
        
    Returns:
        Dictionary with complete path analysis
    """
    components = compute_reward_components(path_edges)
    
    analysis = {
        'path_length': len(path_edges),
        'nodes_visited': len(path_nodes) if path_nodes else len(path_edges) + 1,
        'total_cost': components['total_cost'],
        'total_co2': components['total_co2'],
        'total_distance': components['total_distance'],
        'total_time': components['total_time'],
        'penalty': components['penalty'],
        'reward': components['reward'],
        'avg_cost_per_edge': components['total_cost'] / len(path_edges) if path_edges else 0,
        'avg_co2_per_edge': components['total_co2'] / len(path_edges) if path_edges else 0,
    }
    
    if path_nodes:
        analysis['path_nodes'] = path_nodes
    
    return analysis


# ==================== DEBUG & VISUALIZATION ====================

def pretty_print_path(path_edges: List[Dict[str, Any]], 
                      path_nodes: List[str] = None,
                      show_details: bool = True) -> None:
    """
    Pretty print path information for debugging
    
    Args:
        path_edges: List of edge dictionaries
        path_nodes: Optional list of node IDs in the path
        show_details: Whether to show detailed edge information
    """
    print("="*80)
    print("PATH ANALYSIS")
    print("="*80)
    
    if path_nodes:
        print(f"\nPath: {' → '.join(path_nodes)}")
    
    print(f"\nPath Length: {len(path_edges)} edges")
    
    if show_details:
        print("\nEdge Details:")
        print("-"*80)
        for i, edge in enumerate(path_edges, 1):
            from_node = path_nodes[i-1] if path_nodes and i-1 < len(path_nodes) else "?"
            to_node = edge.get('to', '?')
            print(f"  Step {i}: {from_node} → {to_node}")
            print(f"    Cost: {edge.get('cost', 0):.2f} | "
                  f"CO2: {edge.get('co2', 0):.2f} kg | "
                  f"Distance: {edge.get('dist', 0):.2f} km | "
                  f"Time: {edge.get('time', 0):.2f} hrs")
            if 'truck_type' in edge:
                print(f"    Truck: {edge['truck_type']} | "
                      f"Capacity: {edge.get('capacity', 'N/A')} kg")
    
    # Summary metrics
    analysis = analyze_path(path_edges, path_nodes)
    
    print("\n" + "-"*80)
    print("SUMMARY METRICS")
    print("-"*80)
    print(f"  Total Cost:     {analysis['total_cost']:>10.2f}")
    print(f"  Total CO2:      {analysis['total_co2']:>10.2f} kg")
    print(f"  Total Distance: {analysis['total_distance']:>10.2f} km")
    print(f"  Total Time:     {analysis['total_time']:>10.2f} hours")
    
    print("\n" + "-"*80)
    print("REWARD CALCULATION")
    print("-"*80)
    print(f"  Cost Weight:    {COST_WEIGHT}")
    print(f"  CO2 Weight:     {CO2_WEIGHT}")
    print(f"  Penalty:        {analysis['penalty']:.6f}")
    print(f"  Reward:         {analysis['reward']:.6f}")
    
    print("\n" + "-"*80)
    print("AVERAGES PER EDGE")
    print("-"*80)
    print(f"  Avg Cost:       {analysis['avg_cost_per_edge']:>10.2f}")
    print(f"  Avg CO2:        {analysis['avg_co2_per_edge']:>10.2f} kg")
    
    print("="*80)


def compare_paths(paths: List[Tuple[List[Dict[str, Any]], List[str], str]]) -> None:
    """
    Compare multiple paths side by side
    
    Args:
        paths: List of tuples (path_edges, path_nodes, path_name)
    """
    print("="*80)
    print("PATH COMPARISON")
    print("="*80)
    
    results = []
    for path_edges, path_nodes, name in paths:
        analysis = analyze_path(path_edges, path_nodes)
        analysis['name'] = name
        results.append(analysis)
    
    # Print comparison table
    print(f"\n{'Path Name':<20} {'Length':<8} {'Cost':<12} {'CO2':<12} {'Reward':<12}")
    print("-"*80)
    
    for res in results:
        print(f"{res['name']:<20} "
              f"{res['path_length']:<8} "
              f"{res['total_cost']:<12.2f} "
              f"{res['total_co2']:<12.2f} "
              f"{res['reward']:<12.6f}")
    
    # Highlight best path
    best_reward_path = max(results, key=lambda x: x['reward'])
    print("\n" + "="*80)
    print(f"Best Path (by reward): {best_reward_path['name']}")
    print(f"  Reward: {best_reward_path['reward']:.6f}")
    print("="*80)


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Reward Functions")
    print("="*80)
    
    # Sample path edges
    sample_path = [
        {'to': 'I1', 'cost': 629.68, 'co2': 94.45, 'dist': 78.71, 'time': 1.75, 'truck_type': 'heavy', 'capacity': 12000},
        {'to': 'I2', 'cost': 73.40, 'co2': 5.50, 'dist': 18.35, 'time': 0.32, 'truck_type': 'small', 'capacity': 3500},
        {'to': 'I3', 'cost': 169.24, 'co2': 12.69, 'dist': 42.31, 'time': 0.73, 'truck_type': 'small', 'capacity': 3500},
        {'to': 'I4', 'cost': 386.04, 'co2': 38.60, 'dist': 64.34, 'time': 1.24, 'truck_type': 'medium', 'capacity': 8000},
        {'to': 'D1', 'cost': 302.72, 'co2': 22.70, 'dist': 75.68, 'time': 1.30, 'truck_type': 'small', 'capacity': 3500},
    ]
    
    sample_nodes = ['S1', 'I1', 'I2', 'I3', 'I4', 'D1']
    
    # Test basic reward computation
    print("\n1. Basic Reward Computation")
    print("-"*80)
    reward = compute_reward(sample_path)
    print(f"Reward: {reward:.6f}")
    
    # Test detailed components
    print("\n2. Detailed Components")
    print("-"*80)
    components = compute_reward_components(sample_path)
    for key, value in components.items():
        print(f"  {key}: {value:.2f}")
    
    # Test pretty print
    print("\n3. Pretty Print Path")
    pretty_print_path(sample_path, sample_nodes, show_details=True)
    
    # Test multi-objective with different weights
    print("\n4. Multi-Objective Rewards")
    print("-"*80)
    scenarios = [
        (COST_WEIGHT, CO2_WEIGHT, 0.0, "Default (Cost + CO2)"),
        (0.001, 0.0, 0.0, "Cost Only"),
        (0.0, 0.01, 0.0, "CO2 Only"),
        (0.001, 0.01, 0.1, "Cost + CO2 + Time"),
    ]
    
    for cost_w, co2_w, time_w, name in scenarios:
        reward = compute_multi_objective_reward(sample_path, cost_w, co2_w, time_w)
        print(f"  {name:<25} Reward: {reward:.6f}")
    
    print("\n" + "="*80)
    print("✓ All reward function tests completed!")
    print("="*80)
