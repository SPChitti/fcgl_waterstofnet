"""
Step 7: Create Graph CSV Files

Converts graph_network.json to CSV format with:
- nodes.csv: Node information (id, type, name, lat, lon)
- edges.csv: Edge information with road features and traffic metadata

These CSVs will be the base for FCGL training.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
GRAPH_DIR = BASE_DIR / "graph_network"
CONFIG_DIR = BASE_DIR / "graph_config"

CONFIG_DIR.mkdir(exist_ok=True)

def extract_edge_features(edge):
    """Extract all features and traffic data for an edge"""
    
    features = edge.get('features', {})
    traffic = edge.get('traffic', {})
    
    # Road features
    speeds = features.get('speeds', [])
    slopes = features.get('slopes', [])
    road_class = features.get('road_class', 'unknown')
    
    # Traffic data
    morning_speeds = traffic.get('morning_speeds', [])
    offpeak_speeds = traffic.get('offpeak_speeds', [])
    
    # Calculate aggregates
    avg_speed = np.mean(speeds) if speeds else 50.0
    max_speed = np.max(speeds) if speeds else 50.0
    min_speed = np.min(speeds) if speeds else 50.0
    
    avg_slope = np.mean([abs(s) for s in slopes]) if slopes else 0.0
    max_slope = np.max([abs(s) for s in slopes]) if slopes else 0.0
    
    # Calculate congestion (speed reduction from free flow)
    morning_congestion = 0.0
    offpeak_congestion = 0.0
    
    if morning_speeds and speeds:
        morning_avg = np.mean(morning_speeds)
        free_flow = np.mean(speeds)
        morning_congestion = max(0, (free_flow - morning_avg) / free_flow * 100)
    
    if offpeak_speeds and speeds:
        offpeak_avg = np.mean(offpeak_speeds)
        free_flow = np.mean(speeds)
        offpeak_congestion = max(0, (free_flow - offpeak_avg) / free_flow * 100)
    
    return {
        'avg_speed_kmh': round(avg_speed, 2),
        'max_speed_kmh': round(max_speed, 2),
        'min_speed_kmh': round(min_speed, 2),
        'avg_slope_pct': round(avg_slope, 2),
        'max_slope_pct': round(max_slope, 2),
        'road_class': road_class,
        'morning_congestion_pct': round(morning_congestion, 2),
        'offpeak_congestion_pct': round(offpeak_congestion, 2),
        'morning_avg_speed_kmh': round(np.mean(morning_speeds), 2) if morning_speeds else avg_speed,
        'offpeak_avg_speed_kmh': round(np.mean(offpeak_speeds), 2) if offpeak_speeds else avg_speed,
    }

def create_csv_files():
    """Convert graph JSON to CSV files"""
    
    print("="*80)
    print("Step 7: Creating Graph CSV Files")
    print("="*80)
    
    # Load graph
    graph_file = GRAPH_DIR / 'graph_network.json'
    print(f"\nLoading graph from {graph_file}...")
    
    with open(graph_file, 'r') as f:
        graph = json.load(f)
    
    nodes = graph['nodes']
    edges = graph['edges']
    
    print(f"  Loaded: {len(nodes)} nodes, {len(edges)} edges")
    
    # ==================== CREATE NODES.CSV ====================
    print("\nCreating nodes.csv...")
    
    nodes_data = []
    for node_id, node in nodes.items():
        nodes_data.append({
            'node_id': int(node_id),
            'node_type': node['type'],
            'node_name': node.get('name', f"Node_{node_id}"),
            'lat': node['lat'],
            'lon': node['lon'],
            'num_od_pairs': len(node.get('od_pairs', [])),
            'od_pairs': ','.join(node.get('od_pairs', []))
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    nodes_df = nodes_df.sort_values('node_id')
    
    nodes_file = CONFIG_DIR / 'nodes.csv'
    nodes_df.to_csv(nodes_file, index=False)
    print(f"  ✓ Saved: {nodes_file}")
    print(f"    Rows: {len(nodes_df)}")
    print(f"    Columns: {list(nodes_df.columns)}")
    
    # ==================== CREATE EDGES.CSV ====================
    print("\nCreating edges.csv...")
    
    edges_data = []
    for edge in edges:
        # Extract base edge info
        edge_info = {
            'edge_id': edge['edge_id'],
            'from_node': edge['from_node'],
            'to_node': edge['to_node'],
            'od_pair': edge['od_pair'],
            'distance_km': round(edge['distance_km'], 3),
            'distance_m': round(edge['distance_m'], 1),
        }
        
        # Extract features and traffic
        features = extract_edge_features(edge)
        edge_info.update(features)
        
        # Calculate travel time (hours) based on average speed
        travel_time_freeflow = edge['distance_km'] / features['avg_speed_kmh']
        travel_time_morning = edge['distance_km'] / features['morning_avg_speed_kmh']
        travel_time_offpeak = edge['distance_km'] / features['offpeak_avg_speed_kmh']
        
        edge_info.update({
            'travel_time_freeflow_hours': round(travel_time_freeflow, 4),
            'travel_time_morning_hours': round(travel_time_morning, 4),
            'travel_time_offpeak_hours': round(travel_time_offpeak, 4),
        })
        
        edges_data.append(edge_info)
    
    edges_df = pd.DataFrame(edges_data)
    edges_df = edges_df.sort_values(['od_pair', 'edge_id'])
    
    edges_file = CONFIG_DIR / 'edges.csv'
    edges_df.to_csv(edges_file, index=False)
    print(f"  ✓ Saved: {edges_file}")
    print(f"    Rows: {len(edges_df)}")
    print(f"    Columns: {list(edges_df.columns)}")
    
    # ==================== STATISTICS ====================
    print("\n" + "="*80)
    print("CSV FILES SUMMARY")
    print("="*80)
    
    print("\nNodes by Type:")
    print(nodes_df['node_type'].value_counts())
    
    print("\nEdges by OD Pair:")
    print(edges_df['od_pair'].value_counts())
    
    print("\nEdge Statistics:")
    print(f"  Distance (km):")
    print(f"    Mean: {edges_df['distance_km'].mean():.2f}")
    print(f"    Min: {edges_df['distance_km'].min():.2f}")
    print(f"    Max: {edges_df['distance_km'].max():.2f}")
    
    print(f"\n  Average Speed (km/h):")
    print(f"    Mean: {edges_df['avg_speed_kmh'].mean():.2f}")
    print(f"    Min: {edges_df['min_speed_kmh'].min():.2f}")
    print(f"    Max: {edges_df['max_speed_kmh'].max():.2f}")
    
    print(f"\n  Slope (%):")
    print(f"    Avg: {edges_df['avg_slope_pct'].mean():.2f}")
    print(f"    Max: {edges_df['max_slope_pct'].max():.2f}")
    
    print(f"\n  Congestion (%):")
    print(f"    Morning avg: {edges_df['morning_congestion_pct'].mean():.2f}")
    print(f"    Off-peak avg: {edges_df['offpeak_congestion_pct'].mean():.2f}")
    
    print(f"\n  Road Classes:")
    print(edges_df['road_class'].value_counts())
    
    print("\n" + "="*80)
    print("✓ CSV files created successfully")
    print("="*80)
    print(f"\nFiles saved in: {CONFIG_DIR}/")
    print(f"  - nodes.csv ({len(nodes_df)} rows)")
    print(f"  - edges.csv ({len(edges_df)} rows)")
    print("\nThese CSVs contain all graph metadata and can be modified for FCGL training.")
    print("="*80)

if __name__ == '__main__':
    create_csv_files()
