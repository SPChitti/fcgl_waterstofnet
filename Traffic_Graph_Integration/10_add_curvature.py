"""
Step 10: Add Curvature Data to Graph Network

Calculates curvature metrics from coordinate sequences:
- Average curvature (mean angle change per km)
- Maximum curvature (sharpest turn)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
GRAPH_FILE = BASE_DIR / "graph_network" / "graph_network.json"
FLOW_EDGES_FILE = BASE_DIR / "graph_config" / "flow_network_edges.csv"
ADJACENCY_FILE = BASE_DIR / "graph_config" / "adjacency_list_multimodal.csv"


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points in degrees"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def calculate_angle_change(bearing1, bearing2):
    """Calculate the smallest angle between two bearings"""
    diff = abs(bearing2 - bearing1)
    if diff > 180:
        diff = 360 - diff
    return diff


def calculate_curvature_from_coordinates(coords):
    """
    Calculate curvature metrics from coordinate sequence
    
    Returns:
        avg_curvature_deg_km: Average angle change per kilometer
        max_curvature_deg: Maximum single angle change (sharpest turn)
    """
    if len(coords) < 3:
        return 0.0, 0.0
    
    angle_changes = []
    distances = []
    
    for i in range(len(coords) - 2):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        lon3, lat3 = coords[i + 2]
        
        # Calculate bearings
        bearing1 = calculate_bearing(lat1, lon1, lat2, lon2)
        bearing2 = calculate_bearing(lat2, lon2, lat3, lon3)
        
        # Calculate angle change
        angle_change = calculate_angle_change(bearing1, bearing2)
        angle_changes.append(angle_change)
        
        # Calculate distance for this segment (haversine)
        R = 6371  # Earth radius in km
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        distances.append(distance)
    
    if not angle_changes:
        return 0.0, 0.0
    
    # Average curvature: total angle change per km
    total_distance = sum(distances)
    total_angle_change = sum(angle_changes)
    
    avg_curvature = total_angle_change / total_distance if total_distance > 0 else 0.0
    max_curvature = max(angle_changes)
    
    return avg_curvature, max_curvature


def add_curvature_to_edges():
    """Add curvature metrics to all edges"""
    
    print("=" * 80)
    print("Step 10: Adding Curvature Data")
    print("=" * 80)
    
    # Load graph network
    print("\n📖 Loading graph network...")
    with open(GRAPH_FILE, 'r') as f:
        graph = json.load(f)
    
    # Calculate curvature for each edge
    print(f"\n🔄 Calculating curvature for {len(graph['edges'])} edges...")
    
    for edge in graph['edges']:
        coords = edge['geometry']  # geometry is already a list of coordinates
        avg_curv, max_curv = calculate_curvature_from_coordinates(coords)
        
        edge['features']['avg_curvature_deg_km'] = round(avg_curv, 2)
        edge['features']['max_curvature_deg'] = round(max_curv, 2)
    
    # Save updated graph
    print("💾 Saving updated graph network...")
    with open(GRAPH_FILE, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print("✓ Graph network updated")
    
    # Update flow_network_edges.csv
    print("\n📊 Updating flow_network_edges.csv...")
    
    # Create edge curvature lookup
    edge_curvature = {}
    for edge in graph['edges']:
        from_node = edge['from_node']
        to_node = edge['to_node']
        avg_curv = edge['features']['avg_curvature_deg_km']
        max_curv = edge['features']['max_curvature_deg']
        
        # Store by (from, to) pair - will aggregate later
        key = (from_node, to_node)
        if key not in edge_curvature:
            edge_curvature[key] = {'avg': [], 'max': []}
        edge_curvature[key]['avg'].append(avg_curv)
        edge_curvature[key]['max'].append(max_curv)
    
    # Aggregate curvature for unique edges
    aggregated_curvature = {}
    for key, values in edge_curvature.items():
        aggregated_curvature[key] = {
            'avg_curvature_deg_km': round(np.mean(values['avg']), 2),
            'max_curvature_deg': round(np.max(values['max']), 2)
        }
    
    # Load flow edges and add curvature
    df_flow = pd.read_csv(FLOW_EDGES_FILE)
    
    df_flow['avg_curvature_deg_km'] = df_flow.apply(
        lambda row: aggregated_curvature.get((row['from_node'], row['to_node']), {})
        .get('avg_curvature_deg_km', 0.0),
        axis=1
    )
    
    df_flow['max_curvature_deg'] = df_flow.apply(
        lambda row: aggregated_curvature.get((row['from_node'], row['to_node']), {})
        .get('max_curvature_deg', 0.0),
        axis=1
    )
    
    # Save updated flow edges
    df_flow.to_csv(FLOW_EDGES_FILE, index=False)
    
    print(f"✓ Added curvature to {len(df_flow)} edges")
    print(f"  Avg curvature range: {df_flow['avg_curvature_deg_km'].min():.2f} - {df_flow['avg_curvature_deg_km'].max():.2f} deg/km")
    print(f"  Max curvature range: {df_flow['max_curvature_deg'].min():.2f} - {df_flow['max_curvature_deg'].max():.2f} degrees")
    
    # Update adjacency list
    print("\n📋 Updating adjacency_list_multimodal.csv...")
    
    df_adj = pd.read_csv(ADJACENCY_FILE)
    
    # Create mapping from flow edges
    curvature_map = df_flow.set_index(['from_node', 'to_node'])[
        ['avg_curvature_deg_km', 'max_curvature_deg']
    ].to_dict('index')
    
    # Add curvature columns
    df_adj['avg_curvature_deg_km'] = df_adj.apply(
        lambda row: curvature_map.get((row['from_node'], row['to_node']), {})
        .get('avg_curvature_deg_km', 0.0),
        axis=1
    )
    
    df_adj['max_curvature_deg'] = df_adj.apply(
        lambda row: curvature_map.get((row['from_node'], row['to_node']), {})
        .get('max_curvature_deg', 0.0),
        axis=1
    )
    
    # Save updated adjacency list
    df_adj.to_csv(ADJACENCY_FILE, index=False)
    
    print(f"✓ Added curvature to {len(df_adj)} rows ({len(df_adj)//3} edges × 3 truck types)")
    
    # Show sample
    print("\n📈 Sample with curvature:")
    sample_cols = ['from_node', 'to_node', 'truck_type', 'distance_km', 
                   'avg_slope_pct', 'avg_curvature_deg_km', 'max_curvature_deg']
    print(df_adj[sample_cols].head(6).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✅ Curvature data added successfully!")
    print("=" * 80)


if __name__ == "__main__":
    add_curvature_to_edges()
