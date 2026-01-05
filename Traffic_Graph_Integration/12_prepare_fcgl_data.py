"""
Step 12: Prepare FCGL Training Data

Adapts graph data for FCGL training:
- Renames nodes to match FCGL conventions (0→S0, 1→S1, 2→D2, etc.)
- Renames columns to match graph_env.py expectations
- Creates FCGL-compatible nodes.csv and adjacency_list_multimodal.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
GRAPH_CONFIG = BASE_DIR / "graph_config"
MODEL_DIR = BASE_DIR / "model"
MODEL_GRAPH_CONFIG = MODEL_DIR / "graph_config"

# Ensure output directory exists
MODEL_GRAPH_CONFIG.mkdir(parents=True, exist_ok=True)


def create_node_mapping():
    """
    Create mapping from numeric node IDs to FCGL format
    
    FCGL expects:
    - Sources: S0, S1, ...
    - Destinations: D2, D3, D4, ...
    - Junctions: Keep as numeric strings "5", "6", ...
    """
    # Load nodes to identify sources and destinations
    nodes_df = pd.read_csv(GRAPH_CONFIG / "nodes.csv")
    
    node_mapping = {}
    
    for _, row in nodes_df.iterrows():
        node_id = row['node_id']
        node_type = row['node_type']
        
        if node_type == 'source':
            # Source nodes: 0→S0, 1→S1
            node_mapping[node_id] = f"S{node_id}"
        elif node_type == 'destination':
            # Destination nodes: 2→D2, 3→D3, 4→D4
            node_mapping[node_id] = f"D{node_id}"
        else:
            # Junctions: keep as string "5", "6", etc.
            node_mapping[node_id] = str(node_id)
    
    return node_mapping


def adapt_nodes_csv(node_mapping):
    """Create FCGL-compatible nodes.csv"""
    
    print("\n📋 Adapting nodes.csv...")
    
    # Load original nodes
    nodes_df = pd.read_csv(GRAPH_CONFIG / "nodes.csv")
    
    # Rename node IDs
    nodes_df['node_id'] = nodes_df['node_id'].map(node_mapping)
    
    # Save to model directory
    output_file = MODEL_GRAPH_CONFIG / "nodes.csv"
    nodes_df.to_csv(output_file, index=False)
    
    print(f"✓ Created {output_file}")
    print(f"  Nodes: {len(nodes_df)}")
    print(f"  Sources: {len(nodes_df[nodes_df['node_type']=='source'])}")
    print(f"  Destinations: {len(nodes_df[nodes_df['node_type']=='destination'])}")
    print(f"  Junctions: {len(nodes_df[nodes_df['node_type']=='junction'])}")
    
    # Show sample
    sources = nodes_df[nodes_df['node_type']=='source']['node_id'].tolist()
    destinations = nodes_df[nodes_df['node_type']=='destination']['node_id'].tolist()
    print(f"  Source IDs: {sources}")
    print(f"  Destination IDs: {destinations}")
    
    return nodes_df


def adapt_adjacency_list(node_mapping):
    """Create FCGL-compatible adjacency_list_multimodal.csv"""
    
    print("\n📊 Adapting adjacency_list_multimodal.csv...")
    
    # Load original adjacency list
    adj_df = pd.read_csv(GRAPH_CONFIG / "adjacency_list_multimodal.csv")
    
    print(f"  Original: {len(adj_df)} rows")
    print(f"  Columns: {list(adj_df.columns)}")
    
    # Rename node IDs
    adj_df['from_node'] = adj_df['from_node'].map(node_mapping)
    adj_df['to_node'] = adj_df['to_node'].map(node_mapping)
    
    # Rename columns to match graph_env.py expectations
    column_rename = {
        'from_node': 'node',
        'to_node': 'neighbor',
        'travel_time_hours': 'time_hours',
        'truck_type': 'mode_type'
    }
    
    adj_df = adj_df.rename(columns=column_rename)
    
    # Add mode column (all "truck" since we only have truck transport)
    adj_df['mode'] = 'truck'
    
    # Reorder columns to match fcgl_poc_3 format
    # Required columns: node, neighbor, mode, mode_type, capacity_kg, cost, co2_kg, time_hours, distance_km
    # Optional: all the road feature columns
    core_columns = [
        'node', 'neighbor', 'mode', 'mode_type', 
        'capacity_kg', 'cost', 'co2_kg', 'time_hours', 'distance_km'
    ]
    
    # Add all other columns (road features)
    other_columns = [col for col in adj_df.columns if col not in core_columns]
    final_columns = core_columns + other_columns
    
    adj_df = adj_df[final_columns]
    
    # Save to model directory
    output_file = MODEL_GRAPH_CONFIG / "adjacency_list_multimodal.csv"
    adj_df.to_csv(output_file, index=False)
    
    print(f"✓ Created {output_file}")
    print(f"  Rows: {len(adj_df)}")
    print(f"  Unique edges: {len(adj_df)//3} (× 3 truck types)")
    print(f"  Columns: {len(adj_df.columns)}")
    
    # Show sample
    print("\n  Sample rows:")
    sample_cols = ['node', 'neighbor', 'mode', 'mode_type', 'capacity_kg', 'cost', 'distance_km']
    print(adj_df[sample_cols].head(6).to_string(index=False))
    
    return adj_df


def main():
    print("=" * 80)
    print("Step 12: Preparing FCGL Training Data")
    print("=" * 80)
    
    # Create node mapping
    print("\n🗺️  Creating node ID mapping...")
    node_mapping = create_node_mapping()
    
    print(f"\n  Mapping examples:")
    print(f"    0 → {node_mapping[0]}")
    print(f"    1 → {node_mapping[1]}")
    print(f"    2 → {node_mapping[2]}")
    print(f"    3 → {node_mapping[3]}")
    print(f"    4 → {node_mapping[4]}")
    print(f"    5 → {node_mapping[5]}")
    
    # Adapt nodes.csv
    nodes_df = adapt_nodes_csv(node_mapping)
    
    # Adapt adjacency_list_multimodal.csv
    adj_df = adapt_adjacency_list(node_mapping)
    
    print("\n" + "=" * 80)
    print("✅ FCGL training data prepared!")
    print("=" * 80)
    print(f"\nOutput location: {MODEL_GRAPH_CONFIG}")
    print(f"  - nodes.csv ({len(nodes_df)} nodes)")
    print(f"  - adjacency_list_multimodal.csv ({len(adj_df)} edges)")
    print("\nReady for training with train_fcgl.py!")


if __name__ == "__main__":
    main()
