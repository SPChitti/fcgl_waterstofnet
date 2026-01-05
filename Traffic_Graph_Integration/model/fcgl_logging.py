"""
FCGL Logging Utilities
Utilities for tracking edge flows, terminal visits, and training metrics
"""

import pandas as pd
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime


# ==================== GLOBAL STATE ====================

# Edge flow tracking: {(src, dst): count}
_edge_flow_counts = defaultdict(int)

# Detailed edge flow data: {(src, dst): [edge_attributes_list]}
_edge_flow_details = defaultdict(list)

# Terminal visit tracking: {terminal_node: count}
_terminal_visit_counts = defaultdict(int)

# Training metrics: list of dicts
_training_metrics = []

# Trajectory history
_trajectory_history = []


# ==================== EDGE FLOW TRACKING ====================

def record_edge_flow(trajectory: Dict[str, Any]):
    """
    Record edge flows from a trajectory
    
    Args:
        trajectory: Dictionary containing 'path_nodes' and 'path_edges'
    """
    global _edge_flow_counts, _edge_flow_details
    
    path_nodes = trajectory.get('path_nodes', [])
    path_edges = trajectory.get('path_edges', [])
    
    if len(path_nodes) < 2 or not path_edges:
        return
    
    # Record each edge in the path
    for i, edge in enumerate(path_edges):
        src_node = path_nodes[i]
        dst_node = edge.get('to', path_nodes[i+1] if i+1 < len(path_nodes) else 'unknown')
        
        edge_key = (src_node, dst_node)
        
        # Increment count
        _edge_flow_counts[edge_key] += 1
        
        # Store detailed edge attributes
        edge_detail = {
            'src': src_node,
            'dst': dst_node,
            'truck_type': edge.get('truck_type', 'N/A'),
            'cost': edge.get('cost', 0.0),
            'co2': edge.get('co2', 0.0),
            'distance': edge.get('dist', 0.0),
            'time': edge.get('time', 0.0),
            'capacity': edge.get('capacity', 0.0)
        }
        _edge_flow_details[edge_key].append(edge_detail)


def get_edge_flow_stats() -> pd.DataFrame:
    """
    Get edge flow statistics as DataFrame
    
    Returns:
        DataFrame with columns: src, dst, count, avg_cost, avg_co2, avg_distance, avg_time
    """
    global _edge_flow_counts, _edge_flow_details
    
    if not _edge_flow_counts:
        return pd.DataFrame()
    
    stats = []
    
    for edge_key, count in _edge_flow_counts.items():
        src, dst = edge_key
        details = _edge_flow_details[edge_key]
        
        # Compute averages
        avg_cost = sum(d['cost'] for d in details) / len(details)
        avg_co2 = sum(d['co2'] for d in details) / len(details)
        avg_distance = sum(d['distance'] for d in details) / len(details)
        avg_time = sum(d['time'] for d in details) / len(details)
        
        # Get most common truck type
        truck_types = [d['truck_type'] for d in details]
        most_common_truck = max(set(truck_types), key=truck_types.count)
        
        stats.append({
            'src': src,
            'dst': dst,
            'count': count,
            'truck_type': most_common_truck,
            'avg_cost': avg_cost,
            'avg_co2': avg_co2,
            'avg_distance': avg_distance,
            'avg_time': avg_time,
            'total_cost': avg_cost * count,
            'total_co2': avg_co2 * count
        })
    
    df = pd.DataFrame(stats)
    df = df.sort_values('count', ascending=False)
    
    return df


def save_edge_flow_csv(filename: str, output_dir: str = "."):
    """
    Save edge flow statistics to CSV
    
    Args:
        filename: Output filename
        output_dir: Output directory
    """
    df = get_edge_flow_stats()
    
    if df.empty:
        print(f"Warning: No edge flow data to save")
        return
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Edge flow stats saved: {filepath} ({len(df)} edges)")


def reset_edge_flow_tracking():
    """Reset edge flow tracking data"""
    global _edge_flow_counts, _edge_flow_details
    _edge_flow_counts.clear()
    _edge_flow_details.clear()


# ==================== TERMINAL VISIT TRACKING ====================

def record_terminal_visit(sink_node: str, trajectory_info: Dict[str, Any] = None):
    """
    Record a terminal visit
    
    Args:
        sink_node: Terminal node ID
        trajectory_info: Optional additional trajectory information
    """
    global _terminal_visit_counts
    
    _terminal_visit_counts[sink_node] += 1
    
    # Store trajectory info if provided
    if trajectory_info:
        _trajectory_history.append({
            'terminal': sink_node,
            'timestamp': datetime.now(),
            **trajectory_info
        })


def get_terminal_stats() -> pd.DataFrame:
    """
    Get terminal visit statistics as DataFrame
    
    Returns:
        DataFrame with columns: terminal, count, percentage
    """
    global _terminal_visit_counts
    
    if not _terminal_visit_counts:
        return pd.DataFrame()
    
    total_visits = sum(_terminal_visit_counts.values())
    
    stats = []
    for terminal, count in _terminal_visit_counts.items():
        percentage = (count / total_visits * 100) if total_visits > 0 else 0.0
        stats.append({
            'terminal': terminal,
            'count': count,
            'percentage': percentage
        })
    
    df = pd.DataFrame(stats)
    df = df.sort_values('count', ascending=False)
    
    return df


def save_terminal_csv(filename: str, output_dir: str = "."):
    """
    Save terminal visit statistics to CSV
    
    Args:
        filename: Output filename
        output_dir: Output directory
    """
    df = get_terminal_stats()
    
    if df.empty:
        print(f"Warning: No terminal visit data to save")
        return
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Terminal stats saved: {filepath} ({len(df)} terminals)")


def print_terminal_distribution():
    """Print terminal visit distribution to console"""
    df = get_terminal_stats()
    
    if df.empty:
        print("No terminal visits recorded")
        return
    
    print("\nTerminal Visit Distribution:")
    print("-" * 50)
    for _, row in df.iterrows():
        bar_length = int(row['percentage'] / 2)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"  {row['terminal']}: {row['count']:4d} visits ({row['percentage']:5.1f}%) {bar}")
    print("-" * 50)


def reset_terminal_tracking():
    """Reset terminal visit tracking data"""
    global _terminal_visit_counts, _trajectory_history
    _terminal_visit_counts.clear()
    _trajectory_history.clear()


# ==================== TRAINING METRICS TRACKING ====================

def save_training_metrics(step: int,
                         loss: float,
                         reward: float,
                         **kwargs):
    """
    Record training metrics
    
    Args:
        step: Training iteration/step
        loss: Training loss value
        reward: Trajectory reward
        **kwargs: Additional metrics (e.g., logZ, cost, co2, etc.)
    """
    global _training_metrics
    
    metrics = {
        'step': step,
        'loss': loss,
        'reward': reward,
        **kwargs
    }
    
    _training_metrics.append(metrics)


def get_training_metrics_df() -> pd.DataFrame:
    """
    Get training metrics as DataFrame
    
    Returns:
        DataFrame with all recorded metrics
    """
    global _training_metrics
    
    if not _training_metrics:
        return pd.DataFrame()
    
    return pd.DataFrame(_training_metrics)


def save_metrics_csv(filename: str, output_dir: str = "."):
    """
    Save training metrics to CSV
    
    Args:
        filename: Output filename (e.g., 'metrics.csv')
        output_dir: Output directory
    """
    df = get_training_metrics_df()
    
    if df.empty:
        print(f"Warning: No training metrics to save")
        return
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Training metrics saved: {filepath} ({len(df)} steps)")


def append_metrics_csv(filename: str,
                      step: int,
                      loss: float,
                      reward: float,
                      output_dir: str = ".",
                      **kwargs):
    """
    Append a single row to metrics CSV (for incremental logging)
    
    Args:
        filename: CSV filename
        step: Training step
        loss: Loss value
        reward: Reward value
        output_dir: Output directory
        **kwargs: Additional metrics
    """
    filepath = os.path.join(output_dir, filename)
    
    # Create metrics dict
    metrics = {
        'step': step,
        'loss': loss,
        'reward': reward,
        **kwargs
    }
    
    # Create DataFrame
    df = pd.DataFrame([metrics])
    
    # Append to file (create if doesn't exist)
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, mode='w', header=True, index=False)


def reset_metrics_tracking():
    """Reset training metrics tracking"""
    global _training_metrics
    _training_metrics.clear()


# ==================== COMBINED UTILITIES ====================

def reset_all_tracking():
    """Reset all tracking data"""
    reset_edge_flow_tracking()
    reset_terminal_tracking()
    reset_metrics_tracking()
    print("✓ All tracking data reset")


def save_all_logs(output_dir: str, step: int):
    """
    Save all logs at once
    
    Args:
        output_dir: Output directory
        step: Current training step
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving logs for step {step}...")
    save_edge_flow_csv(f"edge_flows_step_{step}.csv", output_dir)
    save_terminal_csv(f"terminal_hist_step_{step}.csv", output_dir)
    save_metrics_csv("metrics.csv", output_dir)


def print_summary_stats():
    """Print summary statistics to console"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY STATISTICS")
    print("="*80)
    
    # Edge flows
    edge_df = get_edge_flow_stats()
    if not edge_df.empty:
        print(f"\nEdge Flows:")
        print(f"  Total unique edges used: {len(edge_df)}")
        print(f"  Total edge traversals: {edge_df['count'].sum()}")
        print(f"  Most used edge: {edge_df.iloc[0]['src']} → {edge_df.iloc[0]['dst']} "
              f"({edge_df.iloc[0]['count']} times)")
    
    # Terminal visits
    terminal_df = get_terminal_stats()
    if not terminal_df.empty:
        print(f"\nTerminal Visits:")
        print(f"  Total terminals reached: {len(terminal_df)}")
        print(f"  Total visits: {terminal_df['count'].sum()}")
        print_terminal_distribution()
    
    # Training metrics
    metrics_df = get_training_metrics_df()
    if not metrics_df.empty:
        print(f"\nTraining Metrics:")
        print(f"  Total steps: {len(metrics_df)}")
        print(f"  Average loss: {metrics_df['loss'].mean():.4f}")
        print(f"  Average reward: {metrics_df['reward'].mean():.4f}")
        if 'total_cost' in metrics_df.columns:
            print(f"  Average cost: {metrics_df['total_cost'].mean():.2f}")
        if 'total_co2' in metrics_df.columns:
            print(f"  Average CO2: {metrics_df['total_co2'].mean():.2f} kg")
    
    print("="*80)


# ==================== ANALYSIS UTILITIES ====================

def get_most_used_edges(top_k: int = 10) -> pd.DataFrame:
    """Get top-k most frequently used edges"""
    df = get_edge_flow_stats()
    return df.head(top_k)


def get_edge_utilization_by_node(node_id: str) -> Dict[str, Any]:
    """Get edge utilization statistics for a specific node"""
    df = get_edge_flow_stats()
    
    outgoing = df[df['src'] == node_id]
    incoming = df[df['dst'] == node_id]
    
    return {
        'node': node_id,
        'outgoing_edges': len(outgoing),
        'incoming_edges': len(incoming),
        'total_outgoing_flow': outgoing['count'].sum() if not outgoing.empty else 0,
        'total_incoming_flow': incoming['count'].sum() if not incoming.empty else 0,
        'outgoing_details': outgoing.to_dict('records'),
        'incoming_details': incoming.to_dict('records')
    }


def get_training_convergence_stats() -> Dict[str, Any]:
    """Analyze training convergence"""
    df = get_training_metrics_df()
    
    if df.empty or len(df) < 100:
        return {'error': 'Insufficient data for convergence analysis'}
    
    # Split into first half and second half
    mid = len(df) // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]
    
    return {
        'total_steps': len(df),
        'first_half_avg_loss': first_half['loss'].mean(),
        'second_half_avg_loss': second_half['loss'].mean(),
        'loss_improvement': first_half['loss'].mean() - second_half['loss'].mean(),
        'first_half_avg_reward': first_half['reward'].mean(),
        'second_half_avg_reward': second_half['reward'].mean(),
        'reward_improvement': second_half['reward'].mean() - first_half['reward'].mean(),
        'final_100_avg_loss': df.tail(100)['loss'].mean(),
        'final_100_avg_reward': df.tail(100)['reward'].mean()
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing FCGL Logging Utilities")
    print("="*80)
    
    # Test edge flow tracking
    print("\n1. Test Edge Flow Tracking")
    print("-"*80)
    
    sample_traj1 = {
        'path_nodes': ['S1', 'I1', 'I2', 'D1'],
        'path_edges': [
            {'to': 'I1', 'cost': 100, 'co2': 10, 'dist': 50, 'time': 1.0, 'truck_type': 'heavy', 'capacity': 12000},
            {'to': 'I2', 'cost': 150, 'co2': 15, 'dist': 75, 'time': 1.5, 'truck_type': 'medium', 'capacity': 8000},
            {'to': 'D1', 'cost': 200, 'co2': 20, 'dist': 100, 'time': 2.0, 'truck_type': 'small', 'capacity': 3500},
        ]
    }
    
    sample_traj2 = {
        'path_nodes': ['S1', 'I1', 'D2'],
        'path_edges': [
            {'to': 'I1', 'cost': 100, 'co2': 10, 'dist': 50, 'time': 1.0, 'truck_type': 'heavy', 'capacity': 12000},
            {'to': 'D2', 'cost': 180, 'co2': 18, 'dist': 90, 'time': 1.8, 'truck_type': 'medium', 'capacity': 8000},
        ]
    }
    
    record_edge_flow(sample_traj1)
    record_edge_flow(sample_traj2)
    
    edge_stats = get_edge_flow_stats()
    print(edge_stats)
    
    # Test terminal tracking
    print("\n2. Test Terminal Visit Tracking")
    print("-"*80)
    
    record_terminal_visit('D1', {'steps': 3, 'cost': 450})
    record_terminal_visit('D2', {'steps': 2, 'cost': 280})
    record_terminal_visit('D1', {'steps': 4, 'cost': 500})
    
    terminal_stats = get_terminal_stats()
    print(terminal_stats)
    print_terminal_distribution()
    
    # Test metrics tracking
    print("\n3. Test Training Metrics Tracking")
    print("-"*80)
    
    for i in range(10):
        save_training_metrics(
            step=i,
            loss=1.0 / (i + 1),
            reward=0.5 + i * 0.05,
            logZ=0.1 * i,
            total_cost=100 + i * 10
        )
    
    metrics_df = get_training_metrics_df()
    print(metrics_df.head())
    
    # Test saving
    print("\n4. Test Saving to CSV")
    print("-"*80)
    
    test_output_dir = "test_logs"
    save_all_logs(test_output_dir, step=1000)
    
    # Test summary
    print("\n5. Test Summary Statistics")
    print_summary_stats()
    
    # Cleanup
    import shutil
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        print(f"\n✓ Cleaned up test directory: {test_output_dir}")
    
    print("\n" + "="*80)
    print("✓ All logging utility tests completed!")
    print("="*80)
