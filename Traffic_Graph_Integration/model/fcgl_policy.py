"""
FCGL Policy Network for Supply Chain Routing
PyTorch neural network policy for action selection with masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


class FCGLPolicy(nn.Module):
    """
    Policy network for FCGL supply chain routing
    
    Receives:
        - current_node (as string, converted to embedding index)
        - remaining_volume (scalar, normalized)
    
    Outputs:
        - Logits over possible actions (outgoing edges)
        - Masked categorical distribution for valid actions
    """
    
    def __init__(self,
                 node_list: List[str],
                 embedding_dim: int = 32,
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 2,
                 max_volume: float = 15000.0):
        """
        Initialize FCGL policy network
        
        Args:
            node_list: List of all node IDs in the graph
            embedding_dim: Dimension of node embeddings
            hidden_dim: Hidden layer dimension
            num_hidden_layers: Number of hidden layers in MLP
            max_volume: Maximum volume for normalization
        """
        super(FCGLPolicy, self).__init__()
        
        self.node_list = node_list
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_volume = max_volume
        
        # Create node to index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        self.num_nodes = len(node_list)
        
        # Node embedding layer
        self.node_embedding = nn.Embedding(
            num_embeddings=self.num_nodes,
            embedding_dim=embedding_dim
        )
        
        # Volume normalization parameters (learnable)
        self.volume_norm_weight = nn.Parameter(torch.ones(1))
        self.volume_norm_bias = nn.Parameter(torch.zeros(1))
        
        # MLP layers
        # Input: node_embedding + normalized_volume
        input_dim = embedding_dim + 1
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer - will be masked based on valid actions
        # We use a large action space and mask invalid actions
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Action projection (dynamic based on number of outgoing edges)
        self.action_head = nn.Linear(hidden_dim, 1)  # Score per action
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        # Xavier initialization for linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
    
    def node_to_index(self, node_id: str) -> int:
        """Convert node ID to embedding index"""
        if node_id not in self.node_to_idx:
            raise ValueError(f"Unknown node: {node_id}")
        return self.node_to_idx[node_id]
    
    def normalize_volume(self, volume: float) -> torch.Tensor:
        """
        Normalize volume with learnable parameters
        
        Args:
            volume: Raw volume value
            
        Returns:
            Normalized volume tensor
        """
        # Simple normalization: volume / max_volume
        normalized = volume / self.max_volume
        
        # Apply learnable affine transformation
        normalized = normalized * self.volume_norm_weight + self.volume_norm_bias
        
        return normalized
    
    def forward(self, node_id: str, volume: float) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            node_id: Current node ID
            volume: Remaining volume
            
        Returns:
            Hidden representation (before action scoring)
        """
        # Get node index and embedding
        node_idx = self.node_to_index(node_id)
        node_idx_tensor = torch.tensor([node_idx], dtype=torch.long)
        node_emb = self.node_embedding(node_idx_tensor)  # (1, embedding_dim)
        
        # Normalize volume
        volume_norm = self.normalize_volume(volume)
        volume_tensor = torch.tensor([[volume_norm]], dtype=torch.float32)
        
        # Concatenate node embedding and volume
        state_repr = torch.cat([node_emb, volume_tensor], dim=1)  # (1, embedding_dim + 1)
        
        # Pass through MLP
        hidden = self.mlp(state_repr)  # (1, hidden_dim)
        
        # Output representation
        output = self.output_layer(hidden)  # (1, hidden_dim)
        
        return output
    
    def compute_action_logits(self, 
                              state_repr: torch.Tensor,
                              outgoing_edges: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute logits for each outgoing edge
        
        Args:
            state_repr: State representation from forward pass
            outgoing_edges: List of possible edge dictionaries
            
        Returns:
            Logits tensor of shape (num_actions,)
        """
        num_actions = len(outgoing_edges)
        
        if num_actions == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Compute score for each action
        # Simple approach: use state representation to score each edge
        logits = []
        
        for edge in outgoing_edges:
            # Score this action
            score = self.action_head(state_repr)  # (1, 1)
            logits.append(score.squeeze())
        
        logits_tensor = torch.stack(logits)  # (num_actions,)
        
        return logits_tensor
    
    def action_distribution(self,
                           state: Tuple[str, float],
                           outgoing_edges: List[Dict[str, Any]]) -> Categorical:
        """
        Create masked categorical distribution over valid actions
        
        Args:
            state: (current_node, remaining_volume)
            outgoing_edges: List of feasible outgoing edges
            
        Returns:
            Categorical distribution over actions
        """
        current_node, remaining_volume = state
        
        # Forward pass to get state representation
        state_repr = self.forward(current_node, remaining_volume)
        
        # Compute logits for each action
        logits = self.compute_action_logits(state_repr, outgoing_edges)
        
        if len(logits) == 0:
            # No valid actions - should not happen in normal flow
            raise ValueError(f"No valid actions from state {state}")
        
        # Apply masking (all provided edges are valid, but we could add capacity filtering)
        mask = torch.ones_like(logits, dtype=torch.bool)
        
        # Check capacity constraints
        for i, edge in enumerate(outgoing_edges):
            if edge.get('capacity', float('inf')) < remaining_volume:
                # Action would exceed capacity - mask it
                mask[i] = False
        
        # Apply mask: set invalid actions to -inf
        masked_logits = logits.clone()
        masked_logits[~mask] = float('-inf')
        
        # Check if any valid actions remain
        if torch.all(~mask):
            # All actions masked - fall back to uniform over all edges
            masked_logits = torch.zeros_like(logits)
        
        # Create categorical distribution
        action_dist = Categorical(logits=masked_logits)
        
        return action_dist
    
    def sample_action(self,
                     state: Tuple[str, float],
                     outgoing_edges: List[Dict[str, Any]],
                     deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy
        
        Args:
            state: (current_node, remaining_volume)
            outgoing_edges: List of feasible outgoing edges
            deterministic: If True, select argmax instead of sampling
            
        Returns:
            Tuple of (action_index, log_probability)
        """
        # Get action distribution
        action_dist = self.action_distribution(state, outgoing_edges)
        
        # Sample or select best action
        if deterministic:
            action_idx = torch.argmax(action_dist.probs)
        else:
            action_idx = action_dist.sample()
        
        # Get log probability
        log_prob = action_dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob
    
    def evaluate_actions(self,
                        state: Tuple[str, float],
                        outgoing_edges: List[Dict[str, Any]],
                        action_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for policy gradient updates
        
        Args:
            state: (current_node, remaining_volume)
            outgoing_edges: List of feasible outgoing edges
            action_indices: Tensor of action indices to evaluate
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        # Get action distribution
        action_dist = self.action_distribution(state, outgoing_edges)
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(action_indices)
        
        # Compute entropy
        entropy = action_dist.entropy()
        
        return log_probs, entropy
    
    def get_action_probs(self,
                        state: Tuple[str, float],
                        outgoing_edges: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Get probability distribution over actions
        
        Args:
            state: (current_node, remaining_volume)
            outgoing_edges: List of feasible outgoing edges
            
        Returns:
            Probability tensor over actions
        """
        action_dist = self.action_distribution(state, outgoing_edges)
        return action_dist.probs


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing FCGL Policy Network")
    print("="*80)
    
    # Create sample node list
    sample_nodes = ['S1', 'S2', 'I1', 'I2', 'I3', 'I4', 'I5', 'D1', 'D2', 'D3']
    
    # Initialize policy
    print("\n1. Initialize Policy Network")
    print("-"*80)
    policy = FCGLPolicy(
        node_list=sample_nodes,
        embedding_dim=32,
        hidden_dim=128,
        num_hidden_layers=2,
        max_volume=15000.0
    )
    print(f"Policy initialized with {sum(p.numel() for p in policy.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
    
    # Test forward pass
    print("\n2. Test Forward Pass")
    print("-"*80)
    state = ('S1', 1000.0)
    state_repr = policy.forward(state[0], state[1])
    print(f"State: {state}")
    print(f"State representation shape: {state_repr.shape}")
    print(f"State representation (first 5 dims): {state_repr[0, :5]}")
    
    # Sample outgoing edges
    outgoing_edges = [
        {'to': 'I1', 'cost': 100.0, 'co2': 10.0, 'dist': 50.0, 'capacity': 12000, 'time': 1.0},
        {'to': 'I2', 'cost': 150.0, 'co2': 15.0, 'dist': 75.0, 'capacity': 8000, 'time': 1.5},
        {'to': 'I3', 'cost': 120.0, 'co2': 12.0, 'dist': 60.0, 'capacity': 3500, 'time': 1.2},
        {'to': 'I4', 'cost': 200.0, 'co2': 20.0, 'dist': 100.0, 'capacity': 12000, 'time': 2.0},
    ]
    
    # Test action distribution
    print("\n3. Test Action Distribution")
    print("-"*80)
    action_dist = policy.action_distribution(state, outgoing_edges)
    probs = action_dist.probs
    print(f"Number of actions: {len(outgoing_edges)}")
    print(f"Action probabilities: {probs}")
    print(f"Sum of probabilities: {probs.sum():.4f}")
    
    # Test action sampling
    print("\n4. Test Action Sampling")
    print("-"*80)
    for i in range(5):
        action_idx, log_prob = policy.sample_action(state, outgoing_edges, deterministic=False)
        selected_edge = outgoing_edges[action_idx]
        print(f"  Sample {i+1}: Action {action_idx} → {selected_edge['to']} "
              f"(log_prob: {log_prob:.4f}, prob: {torch.exp(log_prob):.4f})")
    
    # Test deterministic action
    print("\n5. Test Deterministic Action")
    print("-"*80)
    action_idx, log_prob = policy.sample_action(state, outgoing_edges, deterministic=True)
    selected_edge = outgoing_edges[action_idx]
    print(f"Best action: {action_idx} → {selected_edge['to']}")
    print(f"Log probability: {log_prob:.4f}")
    
    # Test capacity masking
    print("\n6. Test Capacity Masking")
    print("-"*80)
    high_volume_state = ('S1', 10000.0)
    action_dist_masked = policy.action_distribution(high_volume_state, outgoing_edges)
    probs_masked = action_dist_masked.probs
    print(f"State with high volume: {high_volume_state}")
    print(f"Action probabilities (with masking):")
    for i, (edge, prob) in enumerate(zip(outgoing_edges, probs_masked)):
        mask_status = "✓" if edge['capacity'] >= high_volume_state[1] else "✗ (masked)"
        print(f"  Action {i} → {edge['to']}: {prob:.4f} "
              f"(capacity: {edge['capacity']}) {mask_status}")
    
    # Test gradient flow
    print("\n7. Test Gradient Flow")
    print("-"*80)
    action_idx_tensor = torch.tensor([0])
    log_probs, entropy = policy.evaluate_actions(state, outgoing_edges, action_idx_tensor)
    loss = -log_probs.mean()
    loss.backward()
    
    print(f"Log probability: {log_probs.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    print(f"Loss: {loss.item():.4f}")
    
    # Check gradients
    has_grad = sum(1 for p in policy.parameters() if p.grad is not None)
    total_params = sum(1 for _ in policy.parameters())
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    
    # Show some gradient magnitudes
    print("\nSample gradient magnitudes:")
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: {grad_norm:.6f}")
            if grad_norm > 0:
                break
    
    print("\n" + "="*80)
    print("✓ All policy network tests completed!")
    print("="*80)
