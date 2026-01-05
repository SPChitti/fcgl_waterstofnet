"""
Step 6: Build Graph Network Structure

Constructs a directed graph from the 6 routes by:
1. Identifying key nodes (sources, destinations, convergence/divergence points)
2. Merging overlapping segments into single edges
3. Assigning edge attributes (distance, speed, slope, traffic)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import folium

BASE_DIR = Path(__file__).parent
ROUTES_DIR = BASE_DIR / "routes"
FEATURES_DIR = BASE_DIR / "road_features"
TRAFFIC_DIR = BASE_DIR / "traffic_mapped"
GRAPH_DIR = BASE_DIR / "graph_network"
MAPS_DIR = BASE_DIR / "maps"

GRAPH_DIR.mkdir(exist_ok=True)

# Thresholds
NODE_PROXIMITY_THRESHOLD = 100  # meters - points within this are same node
SIMPLIFICATION_THRESHOLD = 500  # meters - simplify segments longer than this


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def load_all_data():
    """Load routes, features, and traffic data"""
    
    data = {}
    
    for route_file in sorted(ROUTES_DIR.glob("S*_to_D*.json")):
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        od_pair = route_data['metadata']['od_pair']
        
        # Load features
        feature_file = FEATURES_DIR / f"{od_pair}_features.json"
        with open(feature_file, 'r') as f:
            features = json.load(f)
        
        # Load traffic
        traffic_file = TRAFFIC_DIR / f"{od_pair}_traffic.json"
        with open(traffic_file, 'r') as f:
            traffic = json.load(f)
        
        data[od_pair] = {
            'route': route_data,
            'features': features,
            'traffic': traffic
        }
    
    return data


def identify_key_nodes(all_data):
    """
    Identify key nodes in the network:
    - Sources (S1, S2)
    - Destinations (D1, D2, D3)
    - Convergence/divergence points (where routes meet/split)
    """
    
    nodes = {}
    node_id = 0
    
    # Add sources
    sources_seen = set()
    for od_pair, data in all_data.items():
        coords = data['route']['metadata']['source_coords']
        source_id = data['route']['metadata']['source_id']
        
        if source_id not in sources_seen:
            nodes[node_id] = {
                'type': 'source',
                'id': source_id,
                'name': data['route']['metadata']['source_name'],
                'lat': coords[0],
                'lon': coords[1],
                'od_pairs': [od_pair]
            }
            sources_seen.add(source_id)
            node_id += 1
        else:
            # Find existing node and add od_pair
            for nid, node in nodes.items():
                if node['type'] == 'source' and node['id'] == source_id:
                    node['od_pairs'].append(od_pair)
    
    # Add destinations
    destinations_seen = set()
    for od_pair, data in all_data.items():
        coords = data['route']['metadata']['destination_coords']
        dest_id = data['route']['metadata']['destination_id']
        
        if dest_id not in destinations_seen:
            nodes[node_id] = {
                'type': 'destination',
                'id': dest_id,
                'name': data['route']['metadata']['destination_name'],
                'lat': coords[0],
                'lon': coords[1],
                'od_pairs': [od_pair]
            }
            destinations_seen.add(dest_id)
            node_id += 1
        else:
            for nid, node in nodes.items():
                if node['type'] == 'destination' and node['id'] == dest_id:
                    node['od_pairs'].append(od_pair)
    
    # Find convergence/divergence points
    # Sample points along routes and find where multiple routes are close
    print("\nIdentifying convergence/divergence points...")
    
    route_samples = {}
    for od_pair, data in all_data.items():
        coords = data['features']['coordinates']
        # Sample every 100th point to speed up
        step = max(1, len(coords) // 100)
        samples = []
        for i in range(0, len(coords), step):
            samples.append({
                'index': i,
                'lat': coords[i][1],
                'lon': coords[i][0],
                'od_pair': od_pair
            })
        route_samples[od_pair] = samples
    
    # Find clusters where multiple routes converge
    convergence_candidates = []
    
    for od_pair1, samples1 in route_samples.items():
        for sample1 in samples1:
            nearby_routes = set([od_pair1])
            
            for od_pair2, samples2 in route_samples.items():
                if od_pair1 >= od_pair2:
                    continue
                
                for sample2 in samples2:
                    dist = haversine_distance(
                        sample1['lat'], sample1['lon'],
                        sample2['lat'], sample2['lon']
                    )
                    
                    if dist < NODE_PROXIMITY_THRESHOLD:
                        nearby_routes.add(od_pair2)
            
            if len(nearby_routes) >= 2:
                convergence_candidates.append({
                    'lat': sample1['lat'],
                    'lon': sample1['lon'],
                    'routes': list(nearby_routes),
                    'num_routes': len(nearby_routes)
                })
    
    # Cluster nearby convergence points
    clustered_convergence = []
    used = set()
    
    for i, cand1 in enumerate(convergence_candidates):
        if i in used:
            continue
        
        cluster_lats = [cand1['lat']]
        cluster_lons = [cand1['lon']]
        cluster_routes = set(cand1['routes'])
        used.add(i)
        
        for j, cand2 in enumerate(convergence_candidates):
            if j <= i or j in used:
                continue
            
            dist = haversine_distance(
                cand1['lat'], cand1['lon'],
                cand2['lat'], cand2['lon']
            )
            
            if dist < NODE_PROXIMITY_THRESHOLD * 2:
                cluster_lats.append(cand2['lat'])
                cluster_lons.append(cand2['lon'])
                cluster_routes.update(cand2['routes'])
                used.add(j)
        
        if len(cluster_routes) >= 2:
            clustered_convergence.append({
                'lat': np.mean(cluster_lats),
                'lon': np.mean(cluster_lons),
                'routes': list(cluster_routes),
                'num_routes': len(cluster_routes)
            })
    
    # Add convergence points as nodes
    for conv in sorted(clustered_convergence, key=lambda x: -x['num_routes']):
        nodes[node_id] = {
            'type': 'convergence',
            'id': f'C{node_id}',
            'name': f'Junction {node_id}',
            'lat': conv['lat'],
            'lon': conv['lon'],
            'od_pairs': conv['routes'],
            'num_routes': conv['num_routes']
        }
        node_id += 1
    
    print(f"  Sources: {len(sources_seen)}")
    print(f"  Destinations: {len(destinations_seen)}")
    print(f"  Convergence points: {len(clustered_convergence)}")
    print(f"  Total nodes: {len(nodes)}")
    
    return nodes


def build_edges(nodes, all_data):
    """
    Build edges between nodes by finding route segments that connect them
    """
    
    print("\nBuilding edges...")
    
    edges = []
    edge_id = 0
    
    # For each route, create edges between consecutive nodes
    for od_pair, data in all_data.items():
        coords = data['features']['coordinates']
        features = data['features']['features']
        traffic_mapping = data['traffic']['traffic_mapping']
        
        # Find which nodes this route passes through
        route_nodes = []
        for node_id, node in nodes.items():
            if od_pair in node['od_pairs']:
                # Special handling for source/destination - they're always endpoints
                if node['type'] == 'source':
                    route_nodes.append({
                        'node_id': node_id,
                        'node': node,
                        'route_index': -1,  # Always first
                        'distance': 0
                    })
                elif node['type'] == 'destination':
                    route_nodes.append({
                        'node_id': node_id,
                        'node': node,
                        'route_index': len(coords),  # Always last
                        'distance': 0
                    })
                else:
                    # Find closest point on route to this node
                    min_dist = float('inf')
                    closest_idx = 0
                    
                    for idx, coord in enumerate(coords):
                        dist = haversine_distance(
                            node['lat'], node['lon'],
                            coord[1], coord[0]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    # Ensure convergence points don't overlap with source (index 0)
                    if closest_idx == 0:
                        closest_idx = 1
                    
                    route_nodes.append({
                        'node_id': node_id,
                        'node': node,
                        'route_index': closest_idx,
                        'distance': min_dist
                    })
        
        # Sort nodes by their position along route
        route_nodes.sort(key=lambda x: x['route_index'])
        
        # Create edges between consecutive nodes
        for i in range(len(route_nodes) - 1):
            from_node = route_nodes[i]
            to_node = route_nodes[i + 1]
            
            start_idx = from_node['route_index']
            end_idx = to_node['route_index']
            
            # Handle source/destination special indices
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(coords) - 1:
                end_idx = len(coords) - 1
            
            if start_idx >= end_idx:
                print(f"  WARNING: Skipping edge {from_node['node_id']} -> {to_node['node_id']} (indices: {start_idx} >= {end_idx})")
                continue
            
            # Extract segment attributes
            segment_coords = coords[start_idx:end_idx+1]
            segment_length = calculate_path_length(segment_coords)
            
            # Get average features for this segment
            segment_features = extract_segment_features(
                features, start_idx, end_idx
            )
            
            # Get average traffic for this segment
            segment_traffic = extract_segment_traffic(
                traffic_mapping, start_idx, end_idx
            )
            
            edges.append({
                'edge_id': edge_id,
                'from_node': from_node['node_id'],
                'to_node': to_node['node_id'],
                'od_pair': od_pair,
                'distance_m': segment_length,
                'distance_km': segment_length / 1000,
                'geometry': segment_coords,
                'features': segment_features,
                'traffic': segment_traffic
            })
            
            edge_id += 1
    
    print(f"  Created {len(edges)} edges")
    
    return edges


def calculate_path_length(coords):
    """Calculate total length of a path"""
    total = 0
    for i in range(len(coords) - 1):
        total += haversine_distance(
            coords[i][1], coords[i][0],
            coords[i+1][1], coords[i+1][0]
        )
    return total


def extract_segment_features(features, start_idx, end_idx):
    """Extract average features for a segment"""
    
    segment_features = {}
    
    for feature_name, feature_data in features.items():
        values = []
        
        for segment in feature_data:
            seg_start = segment['start_index']
            seg_end = segment['end_index']
            
            # Check if this feature segment overlaps with our route segment
            if seg_start <= end_idx and seg_end >= start_idx:
                if segment['value'] is not None:
                    values.append(segment['value'])
        
        if values:
            if feature_name in ['max_speed', 'average_speed']:
                segment_features[feature_name] = np.mean(values)
            elif feature_name in ['average_slope', 'max_slope']:
                segment_features[feature_name] = np.mean(values)
            elif feature_name == 'road_class':
                # Most common road class
                segment_features[feature_name] = max(set(values), key=values.count)
            elif feature_name == 'surface':
                segment_features[feature_name] = max(set(values), key=values.count)
            elif feature_name == 'road_environment':
                segment_features[feature_name] = max(set(values), key=values.count)
    
    return segment_features


def extract_segment_traffic(traffic_mapping, start_idx, end_idx):
    """Extract average traffic metrics for a segment"""
    
    morning_congestion = []
    offpeak_congestion = []
    morning_speed_ratio = []
    offpeak_speed_ratio = []
    
    for idx in range(start_idx, min(end_idx + 1, len(traffic_mapping))):
        mapping = traffic_mapping[idx]
        
        if mapping['traffic']:
            if 'morning_peak' in mapping['traffic']:
                morning_congestion.append(mapping['traffic']['morning_peak']['congestion_factor'])
                morning_speed_ratio.append(mapping['traffic']['morning_peak']['speed_ratio'])
            
            if 'off_peak' in mapping['traffic']:
                offpeak_congestion.append(mapping['traffic']['off_peak']['congestion_factor'])
                offpeak_speed_ratio.append(mapping['traffic']['off_peak']['speed_ratio'])
    
    traffic = {}
    
    if morning_congestion:
        traffic['morning_peak'] = {
            'congestion_factor': np.mean(morning_congestion),
            'speed_ratio': np.mean(morning_speed_ratio)
        }
    
    if offpeak_congestion:
        traffic['off_peak'] = {
            'congestion_factor': np.mean(offpeak_congestion),
            'speed_ratio': np.mean(offpeak_speed_ratio)
        }
    
    return traffic


def save_graph(nodes, edges):
    """Save graph structure to JSON"""
    
    graph = {
        'metadata': {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'num_sources': sum(1 for n in nodes.values() if n['type'] == 'source'),
            'num_destinations': sum(1 for n in nodes.values() if n['type'] == 'destination'),
            'num_convergence': sum(1 for n in nodes.values() if n['type'] == 'convergence')
        },
        'nodes': nodes,
        'edges': edges
    }
    
    output_file = GRAPH_DIR / "graph_network.json"
    with open(output_file, 'w') as f:
        json.dump(graph, f, indent=2)
    
    print(f"\n✓ Graph saved: {output_file}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    
    return output_file


def visualize_graph(nodes, edges):
    """Create visualization of the graph network"""
    
    print("\nCreating graph visualization...")
    
    m = folium.Map(
        location=[50.85, 4.35],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50px; width: 350px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:16px;
                padding: 10px; text-align: center;">
    <b>Graph Network Structure</b><br>
    <small>Nodes and Edges for FCGL Training</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add edges
    for edge in edges:
        coords_folium = [[c[1], c[0]] for c in edge['geometry']]
        
        folium.PolyLine(
            coords_folium,
            color='#3498db',
            weight=3,
            opacity=0.6,
            popup=f"Edge {edge['edge_id']}<br>Distance: {edge['distance_km']:.1f} km<br>Speed: {edge['features'].get('max_speed', 'N/A')} km/h"
        ).add_to(m)
    
    # Add nodes
    for node_id, node in nodes.items():
        if node['type'] == 'source':
            color = 'green'
            icon = 'play'
        elif node['type'] == 'destination':
            color = 'red'
            icon = 'flag'
        else:
            color = 'blue'
            icon = 'circle'
        
        folium.Marker(
            [node['lat'], node['lon']],
            popup=f"<b>{node['name']}</b><br>Type: {node['type']}<br>Routes: {len(node['od_pairs'])}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    # Save
    output_file = MAPS_DIR / "graph_network.html"
    m.save(str(output_file))
    
    print(f"✓ Map saved: {output_file}")


def main():
    print("=" * 80)
    print("Step 6: Graph Network Construction")
    print("=" * 80)
    
    # Load data
    print("\nLoading route, feature, and traffic data...")
    all_data = load_all_data()
    print(f"  ✓ Loaded {len(all_data)} routes")
    
    # Identify nodes
    nodes = identify_key_nodes(all_data)
    
    # Build edges
    edges = build_edges(nodes, all_data)
    
    # Save graph
    save_graph(nodes, edges)
    
    # Visualize
    visualize_graph(nodes, edges)
    
    print("\n" + "=" * 80)
    print("✓ Graph construction complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
