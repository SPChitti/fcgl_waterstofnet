"""
FCGL Training Loop for Supply Chain Routing
Flow-based Constrained Graph Learning with GFlowNet training
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from typing import List, Dict, Tuple, Any
import random
from datetime import datetime

from graph_env import GraphEnvironment
from fcgl_env import FCGLSupplyChainEnv
from fcgl_policy import FCGLPolicy
from reward_functions import compute_reward, compute_reward_components, pretty_print_path
import fcgl_logging


# ==================== CONFIGURATION ====================

class Config:
    """Training configuration"""
    # Paths
    GRAPH_DATA_PATH = "graph_config/adjacency_list_multimodal.csv"
    NODES_PATH = "graph_config/nodes.csv"
    OUTPUT_DIR = "training_outputs"
    
    # Training parameters
    NUM_ITERATIONS = 3000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1  # Number of trajectories per update
    
    # Policy network parameters
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 128
    NUM_HIDDEN_LAYERS = 2
    MAX_VOLUME = 15000.0
    
    # Environment parameters
    DEFAULT_VOLUME = 1000.0
    
    # GFlowNet parameters
    LOGZ_INIT = 0.0  # Initial log partition function estimate
    LOGZ_LR = 0.1    # Learning rate for logZ updates
    
    # Logging parameters
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 500
    
    # Reward parameters (from reward_functions.py)
    COST_WEIGHT = 0.001
    CO2_WEIGHT = 0.01


# ==================== HELPER FUNCTIONS ====================

def create_output_directory(output_dir: str) -> str:
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_graph_and_nodes(graph_path: str, nodes_path: str) -> Tuple[GraphEnvironment, List[str], List[str]]:
    """
    Load graph environment and extract source/terminal nodes
    
    Returns:
        Tuple of (graph_env, source_nodes, terminal_nodes)
    """
    # Load graph environment
    graph_env = GraphEnvironment(graph_path)
    
    # Load nodes to get all node IDs
    nodes_df = pd.read_csv(nodes_path)
    all_nodes = nodes_df['node_id'].tolist()
    
    # Extract sources and terminals
    source_nodes = [n for n in all_nodes if n.startswith('S')]
    terminal_nodes = [n for n in all_nodes if n.startswith('D')]
    
    return graph_env, all_nodes, source_nodes, terminal_nodes


def sample_trajectory(env: FCGLSupplyChainEnv,
                     policy: FCGLPolicy,
                     start_node: str,
                     volume: float,
                     deterministic: bool = False) -> Dict[str, Any]:
    """
    Sample a complete trajectory from policy
    
    Returns:
        Dictionary containing:
            - trajectory: List of (state, action_dict, next_state, log_prob)
            - path_nodes: List of node IDs visited
            - path_edges: List of edge dictionaries
            - terminal_reached: Whether terminal was reached
            - total_steps: Number of steps taken
    """
    # Reset environment
    state = env.reset(start_node, volume)
    
    trajectory = []
    path_nodes = [start_node]
    path_edges = []
    log_probs = []
    
    max_steps = 100  # Safety limit
    
    for step in range(max_steps):
        # Check if terminal
        if env.is_terminal(state):
            break
        
        # Get available actions
        current_node, remaining_volume = state
        outgoing_edges = env.actions(state)
        
        if not outgoing_edges:
            # Dead end - should not happen in well-formed graph
            break
        
        # Sample action from policy
        action_idx, log_prob = policy.sample_action(state, outgoing_edges, deterministic)
        selected_edge = outgoing_edges[action_idx]
        
        # Take step in environment
        next_state, edge_cost, is_terminal = env.step(state, selected_edge)
        
        # Record trajectory
        trajectory.append({
            'state': state,
            'action_idx': action_idx,
            'action_edge': selected_edge,
            'next_state': next_state,
            'log_prob': log_prob,
            'edge_cost': edge_cost,
            'is_terminal': is_terminal
        })
        
        path_nodes.append(next_state[0])
        path_edges.append(selected_edge)
        log_probs.append(log_prob)
        
        # Update state
        state = next_state
    
    terminal_reached = env.is_terminal(state)
    
    return {
        'trajectory': trajectory,
        'path_nodes': path_nodes,
        'path_edges': path_edges,
        'log_probs': log_probs,
        'terminal_reached': terminal_reached,
        'terminal_node': state[0],
        'total_steps': len(trajectory)
    }


def compute_trajectory_loss(log_probs: List[torch.Tensor],
                           reward: float,
                           logZ: torch.Tensor) -> torch.Tensor:
    """
    Compute GFlowNet trajectory balance loss
    
    Loss = (sum(log_probs) + logZ - log(reward))^2
    
    Args:
        log_probs: List of log probabilities for each action
        reward: Trajectory reward (positive value)
        logZ: Log partition function estimate
        
    Returns:
        Trajectory balance loss
    """
    # Sum log probabilities
    log_prob_trajectory = torch.stack(log_probs).sum()
    
    # Log reward (ensure reward is positive and non-zero)
    log_reward = torch.log(torch.tensor(max(reward, 1e-10), dtype=torch.float32))
    
    # Trajectory balance: log P(trajectory) + logZ = log R(trajectory)
    # Loss = (log P + logZ - log R)^2
    balance_error = log_prob_trajectory + logZ - log_reward
    loss = balance_error ** 2
    
    return loss


def save_trajectory_data(trajectory_data: Dict[str, Any],
                        iteration: int,
                        output_dir: str):
    """Save trajectory information to CSV files"""
    
    # Save edge flows
    path_edges = trajectory_data['path_edges']
    if path_edges:
        edges_df = pd.DataFrame([
            {
                'iteration': iteration,
                'step': i,
                'from': trajectory_data['path_nodes'][i],
                'to': edge['to'],
                'truck_type': edge.get('truck_type', 'N/A'),
                'cost': edge.get('cost', 0),
                'co2': edge.get('co2', 0),
                'distance': edge.get('dist', 0),
                'time': edge.get('time', 0),
                'capacity': edge.get('capacity', 0)
            }
            for i, edge in enumerate(path_edges)
        ])
        
        edges_file = os.path.join(output_dir, f"edge_flows_step_{iteration}.csv")
        edges_df.to_csv(edges_file, index=False)
    
    # Save terminal information
    terminal_df = pd.DataFrame([{
        'iteration': iteration,
        'terminal': trajectory_data['terminal_node'],
        'reached': trajectory_data['terminal_reached'],
        'steps': trajectory_data['total_steps']
    }])
    
    terminal_file = os.path.join(output_dir, f"terminal_hist_step_{iteration}.csv")
    terminal_df.to_csv(terminal_file, index=False)


def save_training_summary(training_history: List[Dict[str, Any]],
                         output_dir: str):
    """Save complete training history to CSV"""
    history_df = pd.DataFrame(training_history)
    history_file = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_file, index=False)
    print(f"\n✓ Training history saved to: {history_file}")


# ==================== MAIN TRAINING LOOP ====================

def train_fcgl(config: Config):
    """Main FCGL training loop"""
    
    print("="*80)
    print("FCGL Training - Supply Chain Routing Optimization")
    print("="*80)
    
    # Create output directory
    output_dir = create_output_directory(config.OUTPUT_DIR)
    print(f"\nOutput directory: {output_dir}")
    
    # Load graph and nodes
    print("\nLoading graph...")
    graph_env, all_nodes, source_nodes, terminal_nodes = load_graph_and_nodes(
        config.GRAPH_DATA_PATH,
        config.NODES_PATH
    )
    print(f"  Nodes: {len(all_nodes)} total, {len(source_nodes)} sources, {len(terminal_nodes)} terminals")
    print(f"  Sources: {source_nodes}")
    print(f"  Terminals: {terminal_nodes}")
    
    # Create FCGL environment
    print("\nInitializing environment...")
    env = FCGLSupplyChainEnv(
        graph=graph_env,
        default_volume=config.DEFAULT_VOLUME
    )
    print(f"  Default volume: {config.DEFAULT_VOLUME} kg")
    
    # Create policy network
    print("\nInitializing policy network...")
    policy = FCGLPolicy(
        node_list=all_nodes,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        max_volume=config.MAX_VOLUME
    )
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy parameters: {num_params:,}")
    print(f"  Embedding dim: {config.EMBEDDING_DIM}, Hidden dim: {config.HIDDEN_DIM}")
    
    # Initialize optimizer
    optimizer = optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
    print(f"\nOptimizer: Adam (lr={config.LEARNING_RATE})")
    
    # Initialize logZ (log partition function)
    logZ = torch.nn.Parameter(torch.tensor(config.LOGZ_INIT, dtype=torch.float32))
    logZ_optimizer = optim.SGD([logZ], lr=config.LOGZ_LR)
    print(f"LogZ initialization: {config.LOGZ_INIT}")
    
    # Training history
    training_history = []
    
    # Terminal visit counter
    terminal_counts = {t: 0 for t in terminal_nodes}
    
    # Reset logging utilities
    fcgl_logging.reset_all_tracking()
    
    print("\n" + "="*80)
    print(f"Starting training for {config.NUM_ITERATIONS} iterations...")
    print("="*80 + "\n")
    
    # Training loop with progress bar
    pbar = tqdm(range(config.NUM_ITERATIONS), desc="Training")
    
    for iteration in pbar:
        # Sample random source node
        start_node = random.choice(source_nodes)
        volume = config.DEFAULT_VOLUME
        
        # Sample trajectory
        traj_data = sample_trajectory(env, policy, start_node, volume, deterministic=False)
        
        # Compute reward
        path_edges = traj_data['path_edges']
        reward = compute_reward(path_edges)
        reward_components = compute_reward_components(path_edges)
        
        # Update terminal counts
        if traj_data['terminal_reached']:
            terminal_counts[traj_data['terminal_node']] += 1
        
        # Record edge flows and terminal visits using logging utilities
        fcgl_logging.record_edge_flow(traj_data)
        if traj_data['terminal_reached']:
            fcgl_logging.record_terminal_visit(
                traj_data['terminal_node'],
                {
                    'iteration': iteration,
                    'steps': traj_data['total_steps'],
                    'cost': reward_components['total_cost'],
                    'co2': reward_components['total_co2'],
                    'reward': reward
                }
            )
        
        # Compute loss
        log_probs = traj_data['log_probs']
        loss = compute_trajectory_loss(log_probs, reward, logZ)
        
        # Backpropagation
        optimizer.zero_grad()
        logZ_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logZ_optimizer.step()
        
        # Compute average edge cost
        avg_edge_cost = reward_components['total_cost'] / len(path_edges) if path_edges else 0.0
        
        # Record history
        history_entry = {
            'iteration': iteration,
            'loss': loss.item(),
            'reward': reward,
            'logZ': logZ.item(),
            'start_node': start_node,
            'terminal_node': traj_data['terminal_node'],
            'terminal_reached': traj_data['terminal_reached'],
            'num_steps': traj_data['total_steps'],
            'total_cost': reward_components['total_cost'],
            'total_co2': reward_components['total_co2'],
            'total_distance': reward_components['total_distance'],
            'total_time': reward_components['total_time'],
            'avg_edge_cost': avg_edge_cost
        }
        training_history.append(history_entry)
        
        # Save training metrics using logging utilities
        fcgl_logging.save_training_metrics(
            step=iteration,
            loss=loss.item(),
            reward=reward,
            logZ=logZ.item(),
            start_node=start_node,
            terminal_node=traj_data['terminal_node'],
            num_steps=traj_data['total_steps'],
            total_cost=reward_components['total_cost'],
            total_co2=reward_components['total_co2'],
            total_distance=reward_components['total_distance'],
            total_time=reward_components['total_time']
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'reward': f"{reward:.4f}",
            'steps': traj_data['total_steps'],
            'terminal': traj_data['terminal_node']
        })
        
        # Logging
        if (iteration + 1) % config.LOG_INTERVAL == 0:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration + 1}/{config.NUM_ITERATIONS}")
            print(f"{'='*80}")
            print(f"  Start: {start_node} → Terminal: {traj_data['terminal_node']}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Reward: {reward:.6f}")
            print(f"  LogZ: {logZ.item():.4f}")
            print(f"  Steps: {traj_data['total_steps']}")
            print(f"  Total Cost: {reward_components['total_cost']:.2f}")
            print(f"  Total CO2: {reward_components['total_co2']:.2f} kg")
            print(f"  Avg Edge Cost: {avg_edge_cost:.2f}")
            
            # Print terminal distribution using logging utilities
            fcgl_logging.print_terminal_distribution()
            
            # Print top-k most used edges
            print(f"\n  Top 5 Most Used Edges:")
            top_edges = fcgl_logging.get_most_used_edges(top_k=5)
            if not top_edges.empty:
                for idx, row in top_edges.iterrows():
                    print(f"    {row['src']} → {row['dst']}: {row['count']} times ")
                    print(f"      (avg cost: {row['avg_cost']:.2f}, avg CO2: {row['avg_co2']:.2f} kg)")
        
        # Save trajectory data periodically
        if (iteration + 1) % config.SAVE_INTERVAL == 0:
            save_trajectory_data(traj_data, iteration + 1, output_dir)
            
            # Save all logs using fcgl_logging utilities
            print(f"\n  Saving logs for iteration {iteration + 1}...")
            fcgl_logging.save_all_logs(output_dir, iteration + 1)
            
            # Save model checkpoint
            checkpoint_path = os.path.join(output_dir, f"policy_checkpoint_{iteration + 1}.pt")
            torch.save({
                'iteration': iteration + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'logZ': logZ.item(),
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    # Print summary statistics using logging utilities
    fcgl_logging.print_summary_stats()
    
    # Print convergence analysis
    print("\nConvergence Analysis:")
    print("-"*80)
    convergence = fcgl_logging.get_training_convergence_stats()
    if 'error' not in convergence:
        print(f"  First half avg loss: {convergence['first_half_avg_loss']:.6f}")
        print(f"  Second half avg loss: {convergence['second_half_avg_loss']:.6f}")
        print(f"  Loss improvement: {convergence['loss_improvement']:.6f}")
        print(f"  First half avg reward: {convergence['first_half_avg_reward']:.6f}")
        print(f"  Second half avg reward: {convergence['second_half_avg_reward']:.6f}")
        print(f"  Reward improvement: {convergence['reward_improvement']:.6f}")
        print(f"  Final 100 steps avg loss: {convergence['final_100_avg_loss']:.6f}")
        print(f"  Final 100 steps avg reward: {convergence['final_100_avg_reward']:.6f}")
    
    # Save final results
    print("\n" + "="*80)
    print("Saving final results...")
    print("="*80)
    
    # Save training history
    save_training_summary(training_history, output_dir)
    
    # Save all logs using fcgl_logging utilities
    print("\nSaving all logs...")
    fcgl_logging.save_all_logs(output_dir, config.NUM_ITERATIONS)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "policy_final.pt")
    torch.save({
        'iteration': config.NUM_ITERATIONS,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logZ': logZ.item(),
        'config': config.__dict__,
        'terminal_counts': terminal_counts
    }, final_model_path)
    print(f"✓ Final model saved: {final_model_path}")
    
    # Save terminal distribution
    terminal_dist_df = pd.DataFrame([
        {'terminal': terminal, 'visits': count, 'percentage': count / sum(terminal_counts.values()) * 100}
        for terminal, count in terminal_counts.items()
    ])
    terminal_dist_path = os.path.join(output_dir, "final_terminal_distribution.csv")
    terminal_dist_df.to_csv(terminal_dist_path, index=False)
    print(f"✓ Terminal distribution saved: {terminal_dist_path}")
    
    # Test final policy with deterministic sampling
    print("\n" + "="*80)
    print("Testing Final Policy (Deterministic)")
    print("="*80)
    
    for source in source_nodes:
        print(f"\nTesting from {source}:")
        test_traj = sample_trajectory(env, policy, source, config.DEFAULT_VOLUME, deterministic=True)
        test_reward = compute_reward(test_traj['path_edges'])
        test_components = compute_reward_components(test_traj['path_edges'])
        
        print(f"  Path: {' → '.join(test_traj['path_nodes'])}")
        print(f"  Steps: {test_traj['total_steps']}")
        print(f"  Reward: {test_reward:.6f}")
        print(f"  Cost: {test_components['total_cost']:.2f}")
        print(f"  CO2: {test_components['total_co2']:.2f} kg")
        print(f"  Distance: {test_components['total_distance']:.2f} km")
        print(f"  Time: {test_components['total_time']:.2f} hours")
    
    print("\n" + "="*80)
    print(f"All outputs saved to: {output_dir}")
    print("="*80)
    
    return policy, training_history, output_dir


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create config
    config = Config()
    
    # Run training
    policy, history, output_dir = train_fcgl(config)
    
    print("\n✓ Training completed successfully!")
