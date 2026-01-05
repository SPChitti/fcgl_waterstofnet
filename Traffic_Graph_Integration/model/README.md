# FCGL Training - Traffic Graph Integration

## Overview
FCGL (Flow-based Constrained Graph Learning) model for hydrogen supply chain routing optimization using real Belgian road network data with traffic, slopes, and curvature.

## Data Preparation Complete ✓

### Graph Network
- **119 nodes**: 2 sources (S0=Genk, S1=Antwerp), 3 destinations (D2=Aalst, D3=Ghent, D4=Bruges), 114 junctions
- **411 edges**: 137 physical road segments × 3 truck types (small/medium/heavy)
- **Real road features**: speeds, slopes, curvature, traffic congestion, road classes

### Files
```
model/
├── graph_config/
│   ├── nodes.csv                      # 119 nodes with S/D naming
│   └── adjacency_list_multimodal.csv  # 411 edges with truck types
├── graph_env.py                       # Graph environment
├── fcgl_env.py                        # RL environment
├── fcgl_policy.py                     # Policy network
├── reward_functions.py                # Multi-objective rewards
├── fcgl_logging.py                    # Training metrics
├── train_fcgl.py                      # Training script
└── training_outputs/                  # Output directory
```

## Training Configuration

### Default Parameters
- **Iterations**: 3000
- **Learning rate**: 0.001
- **Embedding dim**: 32
- **Hidden dim**: 128
- **Max volume**: 15000 kg
- **Default volume**: 1000 kg

### Reward Function
- Multi-objective: Cost + CO2 + Time
- Formula: `reward = exp(-(0.05*cost + 0.01*co2 + 0.05*time))`

### Training Approach
- Randomly samples source nodes (S0 or S1)
- Policy learns to route to destinations (D2, D3, D4)
- GFlowNet ensures exploration proportional to rewards
- Learns optimal truck type selection per edge

## Running Training

```bash
cd /home/ubuntu/fcgl_waterstofnet/Traffic_Graph_Integration/model
python3 train_fcgl.py
```

## Outputs
- `training_outputs/run_TIMESTAMP/`
  - `policy_final.pt` - Trained model weights
  - `training_history.csv` - Loss/reward per iteration
  - `edge_flows/` - Which edges were used
  - `terminal_distribution.csv` - Destination visit frequencies
  - Checkpoints every 500 iterations

## Next Steps
1. Train model: `python3 train_fcgl.py`
2. Evaluate on test scenarios
3. Compare against shortest path baselines
4. Visualize learned routes
