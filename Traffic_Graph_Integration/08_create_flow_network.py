"""
Step 8: Create Flow Network and Visualization

Converts graph CSVs to flow network by adding:
- Edge capacities (based on road class and lanes)
- Edge costs (based on distance, time, congestion)
- Source supplies and destination demands
- Visualization with edge metadata
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "graph_config"
MAPS_DIR = BASE_DIR / "maps"

def add_flow_properties():
    """Add flow network properties to edges"""
    
    print("="*80)
    print("Step 8: Creating Flow Network")
    print("="*80)
    
    # Load CSVs
    print("\nLoading graph CSVs...")
    nodes_df = pd.read_csv(CONFIG_DIR / 'nodes.csv')
    edges_df = pd.read_csv(CONFIG_DIR / 'edges.csv')
    print(f"  Loaded: {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    # ==================== ADD EDGE CAPACITIES ====================
    print("\nAdding edge capacities based on road class...")
    
    # Capacity based on road class (vehicles per hour per lane)
    capacity_map = {
        'motorway': 2000,
        'trunk': 1800,
        'primary': 1500,
        'secondary': 1200,
        'tertiary': 800,
        'residential': 400,
        'service': 200,
        'living_street': 100,
        'unknown': 1000
    }
    
    # Assume lanes based on road class
    lanes_map = {
        'motorway': 2,
        'trunk': 2,
        'primary': 2,
        'secondary': 1,
        'tertiary': 1,
        'residential': 1,
        'service': 1,
        'living_street': 1,
        'unknown': 1
    }
    
    edges_df['lanes'] = edges_df['road_class'].map(lanes_map)
    edges_df['capacity_vehicles_per_hour'] = edges_df['road_class'].map(capacity_map) * edges_df['lanes']
    
    # Hydrogen truck capacity (kg per truck)
    # Assume heavy trucks carrying 1000 kg hydrogen each
    kg_per_truck = 1000
    edges_df['capacity_kg_per_hour'] = edges_df['capacity_vehicles_per_hour'] * kg_per_truck
    
    # ==================== ADD EDGE COSTS ====================
    print("Calculating edge costs...")
    
    # Base cost: distance-based (€ per km)
    cost_per_km = 1.5  # € per km for hydrogen truck
    edges_df['distance_cost'] = edges_df['distance_km'] * cost_per_km
    
    # Time cost: based on travel time (€ per hour)
    cost_per_hour = 50.0  # € per hour (driver + vehicle)
    edges_df['time_cost_freeflow'] = edges_df['travel_time_freeflow_hours'] * cost_per_hour
    edges_df['time_cost_morning'] = edges_df['travel_time_morning_hours'] * cost_per_hour
    
    # Congestion penalty
    edges_df['congestion_cost'] = edges_df['morning_congestion_pct'] * 0.1  # € penalty per % congestion
    
    # Total cost (use morning peak for conservative estimate)
    edges_df['total_cost'] = edges_df['distance_cost'] + edges_df['time_cost_morning'] + edges_df['congestion_cost']
    
    # ==================== AGGREGATE EDGES ====================
    print("\nAggregating multi-commodity edges...")
    
    # For flow network, aggregate duplicate (from,to) pairs
    edges_agg = edges_df.groupby(['from_node', 'to_node']).agg({
        'distance_km': 'mean',
        'distance_m': 'mean',
        'avg_speed_kmh': 'mean',
        'max_speed_kmh': 'max',
        'min_speed_kmh': 'min',
        'avg_slope_pct': 'mean',
        'max_slope_pct': 'max',
        'morning_congestion_pct': 'mean',
        'offpeak_congestion_pct': 'mean',
        'morning_avg_speed_kmh': 'mean',
        'offpeak_avg_speed_kmh': 'mean',
        'travel_time_freeflow_hours': 'mean',
        'travel_time_morning_hours': 'mean',
        'travel_time_offpeak_hours': 'mean',
        'capacity_kg_per_hour': 'first',
        'total_cost': 'mean',
        'road_class': 'first',
        'lanes': 'first',
        'od_pair': lambda x: ','.join(sorted(set(x)))  # List all OD pairs
    }).reset_index()
    
    # Add edge ID
    edges_agg.insert(0, 'edge_id', range(len(edges_agg)))
    edges_agg['num_od_pairs'] = edges_agg['od_pair'].str.count(',') + 1
    
    print(f"  Aggregated to {len(edges_agg)} unique edges")
    
    # Save flow network edges
    flow_edges_file = CONFIG_DIR / 'flow_network_edges.csv'
    edges_agg.to_csv(flow_edges_file, index=False)
    print(f"  ✓ Saved: {flow_edges_file}")
    
    # ==================== CREATE DEMAND DATA ====================
    print("\nCreating supply/demand data...")
    
    # Sources (supply nodes)
    sources = nodes_df[nodes_df['node_type'] == 'source'].copy()
    sources['supply_kg_per_hour'] = [5000, 8000]  # Genk: 5000 kg/h, Antwerp: 8000 kg/h
    
    # Destinations (demand nodes)
    destinations = nodes_df[nodes_df['node_type'] == 'destination'].copy()
    destinations['demand_kg_per_hour'] = [4000, 5000, 4000]  # Aalst: 4000, Ghent: 5000, Bruges: 4000
    
    # Create demand CSV
    demand_data = []
    for _, src in sources.iterrows():
        for _, dst in destinations.iterrows():
            od_pair = f"S{src['node_id']}_to_D{dst['node_id']}"
            demand_data.append({
                'od_pair': od_pair,
                'source_node': src['node_id'],
                'source_name': src['node_name'],
                'destination_node': dst['node_id'],
                'destination_name': dst['node_name'],
                'demand_kg_per_hour': dst['demand_kg_per_hour']
            })
    
    demand_df = pd.DataFrame(demand_data)
    demand_file = CONFIG_DIR / 'demand.csv'
    demand_df.to_csv(demand_file, index=False)
    print(f"  ✓ Saved: {demand_file}")
    
    # ==================== STATISTICS ====================
    print("\n" + "="*80)
    print("FLOW NETWORK SUMMARY")
    print("="*80)
    
    print(f"\nNodes: {len(nodes_df)}")
    print(f"  Sources: {len(sources)} (Total supply: {sources['supply_kg_per_hour'].sum()} kg/h)")
    print(f"  Destinations: {len(destinations)} (Total demand: {destinations['demand_kg_per_hour'].sum()} kg/h)")
    print(f"  Junctions: {len(nodes_df) - len(sources) - len(destinations)}")
    
    print(f"\nEdges: {len(edges_agg)}")
    print(f"  Avg capacity: {edges_agg['capacity_kg_per_hour'].mean():.0f} kg/h")
    print(f"  Avg cost: €{edges_agg['total_cost'].mean():.2f}")
    print(f"  Avg distance: {edges_agg['distance_km'].mean():.2f} km")
    
    print(f"\nOD Pairs: {len(demand_df)}")
    for _, row in demand_df.iterrows():
        print(f"  {row['od_pair']}: {row['demand_kg_per_hour']} kg/h")
    
    return nodes_df, edges_agg, demand_df

def visualize_flow_network(nodes_df, edges_df):
    """Create visualization with edge metadata"""
    
    print("\n" + "="*80)
    print("Creating Flow Network Visualization")
    print("="*80)
    
    # Build graph
    G = nx.DiGraph()
    
    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'], **node.to_dict())
    
    for _, edge in edges_df.iterrows():
        G.add_edge(edge['from_node'], edge['to_node'], **edge.to_dict())
    
    # Compute layers
    source_nodes = sorted(nodes_df[nodes_df['node_type'] == 'source']['node_id'].tolist())
    dest_nodes = sorted(nodes_df[nodes_df['node_type'] == 'destination']['node_id'].tolist())
    junction_nodes = sorted(nodes_df[nodes_df['node_type'] == 'convergence']['node_id'].tolist())
    
    layers = {}
    for src in source_nodes:
        layers[src] = 0
    
    queue = [(src, 0) for src in source_nodes]
    visited = set(source_nodes)
    
    while queue:
        current, dist = queue.pop(0)
        for neighbor in G.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                layers[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
            else:
                if neighbor not in dest_nodes:
                    layers[neighbor] = min(layers.get(neighbor, 999), dist + 1)
    
    # Force all destinations to same level
    max_dest_level = max(layers[d] for d in dest_nodes)
    for dest in dest_nodes:
        layers[dest] = max_dest_level
    
    # Position nodes
    layer_groups = defaultdict(list)
    for node_id, layer in layers.items():
        layer_groups[layer].append(node_id)
    
    max_nodes_in_layer = max(len(nodes_list) for nodes_list in layer_groups.values())
    max_level = max(layers.values())
    
    pos = {}
    for layer in range(max_level + 1):
        nodes_in_layer = sorted(layer_groups.get(layer, []))
        num_nodes = len(nodes_in_layer)
        
        if num_nodes == 0:
            continue
        
        x = layer
        y_spacing = 1.0
        total_height = (max_nodes_in_layer - 1) * y_spacing
        column_height = (num_nodes - 1) * y_spacing
        y_offset = (total_height - column_height) / 2
        
        y_positions = [y_offset + i * y_spacing for i in range(num_nodes)]
        
        for node_id, y in zip(nodes_in_layer, y_positions):
            pos[node_id] = (x, -y)
    
    # Create figure
    fig_width = max(24, max_level * 0.8)
    fig_height = 18
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#f9f9f9')
    
    # Draw edges with labels
    print("Drawing edges with metadata...")
    nx.draw_networkx_edges(G, pos,
                          edge_color='#5a6978',
                          width=1.5,
                          alpha=0.5,
                          arrows=True,
                          arrowsize=10,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.0',
                          ax=ax)
    
    # Draw edge labels (show road class, speed, and distance for ALL edges)
    edge_labels = {}
    
    for u, v in G.edges():
        edge_data = G[u][v]
        road_class = edge_data.get('road_class', 'unk')
        avg_speed = edge_data.get('avg_speed_kmh', 0)
        distance = edge_data.get('distance_km', 0)
        congestion = edge_data.get('morning_congestion_pct', 0)
        
        # Compact label: road_class, speed, distance
        label = f"{road_class[:3]}\n{avg_speed:.0f}km/h\n{distance:.1f}km"
        if congestion > 5:
            label += f"\n⚠{congestion:.0f}%"
        edge_labels[(u, v)] = label
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                 font_size=5,
                                 font_color='#2c3e50',
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5),
                                 ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes,
                          node_color='#27ae60',
                          node_size=800,
                          node_shape='s',
                          edgecolors='#1e8449',
                          linewidths=3,
                          ax=ax,
                          label='Sources')
    
    nx.draw_networkx_nodes(G, pos, nodelist=dest_nodes,
                          node_color='#e74c3c',
                          node_size=800,
                          node_shape='s',
                          edgecolors='#c0392b',
                          linewidths=3,
                          ax=ax,
                          label='Destinations')
    
    nx.draw_networkx_nodes(G, pos, nodelist=junction_nodes,
                          node_color='#5dade2',
                          node_size=150,
                          node_shape='o',
                          edgecolors='#2874a6',
                          linewidths=1,
                          alpha=0.8,
                          ax=ax,
                          label='Junctions')
    
    # Node labels
    labels = {}
    for node_id in source_nodes + dest_nodes:
        node_data = nodes_df[nodes_df['node_id'] == node_id].iloc[0]
        labels[node_id] = node_data['node_name'].split()[0]  # First word only
    
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=10,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # Title
    ax.set_title('Flow Network: Belgium Hydrogen Supply Chain\n' +
                 f'{len(source_nodes)} Sources → {len(dest_nodes)} Destinations via {len(junction_nodes)} Junctions\n' +
                 f'{G.number_of_edges()} edges with road features, traffic, and capacities',
                 fontsize=20, fontweight='bold', pad=25)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.set_xlabel('Network Level', fontsize=12)
    ax.margins(0.02)
    ax.grid(True, alpha=0.1)
    ax.axis('off')
    
    plt.tight_layout()
    
    output_path = MAPS_DIR / 'flow_network_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    print(f"  (All {len(edge_labels)} edges labeled with: road_class, speed, distance, congestion)")

def main():
    # Create flow network
    nodes_df, edges_df, demand_df = add_flow_properties()
    
    # Visualize
    visualize_flow_network(nodes_df, edges_df)
    
    print("\n" + "="*80)
    print("✓ Flow Network Created Successfully")
    print("="*80)
    print("\nFiles created in graph_config/:")
    print("  - flow_network_edges.csv (aggregated edges with capacities & costs)")
    print("  - demand.csv (OD pair demands)")
    print("\nVisualization saved in maps/:")
    print("  - flow_network_visualization.png")
    print("="*80)

if __name__ == '__main__':
    main()
