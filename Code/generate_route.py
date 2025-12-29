#!/usr/bin/env python3
"""
Generate detailed routes with elevation, slopes, and road attributes.
Creates HTML maps with node markers every 10 miles for graph creation.
"""

import requests
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
import folium
from folium import plugins
import pandas as pd
from datetime import datetime

GRAPHHOPPER_URL = "http://localhost:8080/route"
MILES_TO_KM = 1.60934
NODE_INTERVAL_MILES = 10
NODE_INTERVAL_KM = NODE_INTERVAL_MILES * MILES_TO_KM

def load_locations(csv_path: str = "../Data/master_locations.csv") -> pd.DataFrame:
    """Load location database."""
    return pd.read_csv(csv_path)

def get_location_coords(location_name: str, locations_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """Get coordinates for a location name."""
    # Try exact match first
    match = locations_df[locations_df['Location'].str.lower() == location_name.lower()]
    if not match.empty:
        return (match.iloc[0]['Latitude'], match.iloc[0]['Longitude'])
    
    # Try partial match
    match = locations_df[locations_df['Location'].str.lower().str.contains(location_name.lower())]
    if not match.empty:
        return (match.iloc[0]['Latitude'], match.iloc[0]['Longitude'])
    
    return None

def query_route_alternatives(
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    profile: str = "truck_diesel",
    num_alternatives: int = 3
) -> List[Dict]:
    """
    Query GraphHopper for multiple alternative routes.
    
    Args:
        start_coords: (lat, lon) of origin
        end_coords: (lat, lon) of destination
        profile: GraphHopper profile name
        num_alternatives: Number of alternative routes (1-5)
    
    Returns:
        List of route dictionaries with full details
    """
    params = {
        "point": [f"{start_coords[0]},{start_coords[1]}", f"{end_coords[0]},{end_coords[1]}"],
        "profile": profile,
        "locale": "en",
        "points_encoded": False,
        "elevation": True,
        "details": [
            "average_slope",
            "max_slope", 
            "road_class",
            "road_access",
            "max_speed",
            "surface",
            "road_environment"
        ],
        "algorithm": "alternative_route",
        "alternative_route.max_paths": num_alternatives,
        "alternative_route.max_weight_factor": 1.6,
        "alternative_route.max_share_factor": 0.7
    }
    
    response = requests.get(GRAPHHOPPER_URL, params=params)
    
    if response.status_code != 200:
        raise Exception(f"GraphHopper API error: {response.status_code}\n{response.text}")
    
    data = response.json()
    
    if "paths" not in data or len(data["paths"]) == 0:
        raise Exception("No routes found")
    
    return data["paths"]

def analyze_route_details(path: Dict) -> Dict:
    """
    Extract detailed statistics from a route path.
    
    Returns:
        Dictionary with route analysis including slopes, elevations, road classes, etc.
    """
    details = path.get("details", {})
    points = path.get("points", {}).get("coordinates", [])
    
    # Basic route info
    distance_km = path.get("distance", 0) / 1000
    distance_miles = distance_km / MILES_TO_KM
    time_min = path.get("time", 0) / (1000 * 60)
    ascend = path.get("ascend", 0)
    descend = path.get("descend", 0)
    
    # Slope analysis
    avg_slopes = details.get("average_slope", [])
    uphill_segments = sum(1 for seg in avg_slopes if seg[2] > 0)
    downhill_segments = sum(1 for seg in avg_slopes if seg[2] < 0)
    steep_segments = sum(1 for seg in avg_slopes if abs(seg[2]) > 5)
    
    max_slopes = details.get("max_slope", [])
    max_uphill = max([seg[2] for seg in max_slopes if seg[2] is not None], default=0)
    max_downhill = min([seg[2] for seg in max_slopes if seg[2] is not None], default=0)
    
    # Road class distribution
    road_classes = details.get("road_class", [])
    class_counts = {}
    for seg in road_classes:
        road_type = seg[2]
        class_counts[road_type] = class_counts.get(road_type, 0) + 1
    
    # Speed analysis
    max_speeds = details.get("max_speed", [])
    valid_speeds = [seg[2] for seg in max_speeds if seg[2] is not None]
    avg_speed_kmh = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0
    
    # Road surface
    surfaces = details.get("surface", [])
    surface_counts = {}
    for seg in surfaces:
        surface_type = seg[2]
        if surface_type:
            surface_counts[surface_type] = surface_counts.get(surface_type, 0) + 1
    
    # Road environment
    environments = details.get("road_environment", [])
    env_counts = {}
    for seg in environments:
        env_type = seg[2]
        if env_type:
            env_counts[env_type] = env_counts.get(env_type, 0) + 1
    
    return {
        "distance_km": round(distance_km, 2),
        "distance_miles": round(distance_miles, 2),
        "duration_min": round(time_min, 1),
        "elevation": {
            "ascent_m": round(ascend, 1),
            "descent_m": round(descend, 1),
            "net_elevation_m": round(ascend - descend, 1)
        },
        "slopes": {
            "uphill_segments": uphill_segments,
            "downhill_segments": downhill_segments,
            "steep_segments_over_5pct": steep_segments,
            "max_uphill_grade_pct": round(max_uphill, 1),
            "max_downhill_grade_pct": round(max_downhill, 1)
        },
        "road_classes": class_counts,
        "surfaces": surface_counts,
        "environments": env_counts,
        "avg_speed_kmh": round(avg_speed_kmh, 1),
        "num_points": len(points)
    }

def create_route_nodes(points: List[List], interval_km: float) -> List[Dict]:
    """
    Create nodes along route at specified intervals for graph creation.
    
    Args:
        points: List of [lon, lat, elevation] coordinates
        interval_km: Distance interval in kilometers
    
    Returns:
        List of node dictionaries with location, lat, lon
    """
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    
    nodes = []
    cumulative_distance = 0
    last_node_distance = 0
    
    geolocator = Nominatim(user_agent="fcgl_route_generator")
    
    # Always add start node
    start_point = points[0]
    try:
        location = geolocator.reverse(f"{start_point[1]}, {start_point[0]}", timeout=10)
        place_name = location.address if location else "Start Point"
    except:
        place_name = "Start Point"
    
    nodes.append({
        "node_id": 0,
        "place_name": place_name,
        "latitude": round(start_point[1], 6),
        "longitude": round(start_point[0], 6),
        "elevation_m": round(start_point[2], 1) if len(start_point) > 2 else 0,
        "distance_from_start_km": 0
    })
    
    # Add intermediate nodes at intervals
    for i in range(1, len(points)):
        prev_point = points[i-1]
        curr_point = points[i]
        
        segment_dist = geodesic(
            (prev_point[1], prev_point[0]),
            (curr_point[1], curr_point[0])
        ).kilometers
        
        cumulative_distance += segment_dist
        
        # Check if we've traveled enough distance for a new node
        if cumulative_distance - last_node_distance >= interval_km:
            try:
                location = geolocator.reverse(f"{curr_point[1]}, {curr_point[0]}", timeout=10)
                place_name = location.address if location else f"Node {len(nodes)}"
            except:
                place_name = f"Node {len(nodes)}"
            
            nodes.append({
                "node_id": len(nodes),
                "place_name": place_name,
                "latitude": round(curr_point[1], 6),
                "longitude": round(curr_point[0], 6),
                "elevation_m": round(curr_point[2], 1) if len(curr_point) > 2 else 0,
                "distance_from_start_km": round(cumulative_distance, 2)
            })
            
            last_node_distance = cumulative_distance
    
    # Always add end node
    end_point = points[-1]
    try:
        location = geolocator.reverse(f"{end_point[1]}, {end_point[0]}", timeout=10)
        place_name = location.address if location else "End Point"
    except:
        place_name = "End Point"
    
    nodes.append({
        "node_id": len(nodes),
        "place_name": place_name,
        "latitude": round(end_point[1], 6),
        "longitude": round(end_point[0], 6),
        "elevation_m": round(end_point[2], 1) if len(end_point) > 2 else 0,
        "distance_from_start_km": round(cumulative_distance, 2)
    })
    
    return nodes

def create_html_map(
    route: Dict,
    start_name: str,
    end_name: str,
    output_path: str
):
    """
    Create interactive HTML map for a SINGLE route with detailed visualizations.
    
    Args:
        route: Single route dictionary with path, analysis, and nodes
        start_name: Origin location name
        end_name: Destination location name
        output_path: Path to save HTML file
    """
    points = route["path"]["points"]["coordinates"]
    analysis = route["analysis"]
    nodes = route.get("nodes", [])
    details = route["path"].get("details", {})
    
    # Get center point
    center_lat = sum(p[1] for p in points) / len(points)
    center_lon = sum(p[0] for p in points) / len(points)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Draw route with color-coded segments (ascent/descent/tunnel)
    # Get slope and environment details
    avg_slopes = details.get("average_slope", [])
    environments = details.get("road_environment", [])
    
    # Create a map of point indices to their properties
    slope_map = {}
    for seg in avg_slopes:
        start_idx, end_idx, slope_val = seg[0], seg[1], seg[2]
        for i in range(start_idx, end_idx):
            slope_map[i] = slope_val
    
    tunnel_map = {}
    for seg in environments:
        start_idx, end_idx, env_type = seg[0], seg[1], seg[2]
        if env_type == "tunnel":
            for i in range(start_idx, end_idx):
                tunnel_map[i] = True
    
    # Draw colored segments
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        coords = [[p1[1], p1[0]], [p2[1], p2[0]]]
        
        # Determine color based on slope and environment
        slope = slope_map.get(i, 0)
        is_tunnel = tunnel_map.get(i, False)
        
        if is_tunnel:
            color = '#7d3c98'  # Dark purple for tunnels
            weight = 10
            tooltip = f"Tunnel (slope: {slope:.1f}%)"
        elif slope > 2:
            color = '#c0392b'  # Dark red for ascent
            weight = 8
            tooltip = f"Ascent: {slope:.1f}%"
        elif slope < -2:
            color = '#1e8449'  # Dark green for descent
            weight = 8
            tooltip = f"Descent: {slope:.1f}%"
        else:
            color = '#2471a3'  # Dark blue for flat
            weight = 7
            tooltip = f"Flat: {slope:.1f}%"
        
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.95,
            tooltip=tooltip
        ).add_to(m)
    
    # Add node markers
    for node in nodes:
        node_popup = f"""
        <b>{node['place_name']}</b><br>
        Node {node['node_id']}<br>
        Distance: {node['distance_from_start_km']} km<br>
        Elevation: {node['elevation_m']} m<br>
        Coordinates: {node['latitude']}, {node['longitude']}
        """
        
        if node['node_id'] == 0:
            icon = folium.Icon(color='green', icon='play', prefix='fa')
        elif node['node_id'] == len(nodes) - 1:
            icon = folium.Icon(color='red', icon='stop', prefix='fa')
        else:
            icon = folium.Icon(color='blue', icon='info-sign')
        
        folium.Marker(
            [node['latitude'], node['longitude']],
            popup=folium.Popup(node_popup, max_width=350),
            icon=icon
        ).add_to(m)
    
    # Build elevation profile data
    elevations = [p[2] if len(p) > 2 else 0 for p in points]
    distances_km = []
    cumulative = 0
    distances_km.append(0)
    for i in range(1, len(points)):
        from geopy.distance import geodesic
        prev = points[i-1]
        curr = points[i]
        segment_dist = geodesic((prev[1], prev[0]), (curr[1], curr[0])).kilometers
        cumulative += segment_dist
        distances_km.append(cumulative)
    
    # Sample elevation data for chart (every 10th point to keep it manageable)
    sample_distances = distances_km[::10]
    sample_elevations = elevations[::10]
    
    # Create charts as HTML
    road_classes = analysis.get('road_classes', {})
    surfaces = analysis.get('surfaces', {})
    environments = analysis.get('environments', {})
    
    # Build comprehensive info panel with charts
    info_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 450px; max-height: 95vh; overflow-y: auto;
                background-color: white; border:2px solid #333; z-index:9999; 
                font-size:13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        
        <h3 style="margin-top:0; color: #2c3e50;">{start_name} → {end_name}</h3>
        <p style="color: #7f8c8d; margin: 5px 0;"><b>Alternative {route['alternative_id']}</b></p>
        <p style="color: #7f8c8d; margin: 5px 0; font-size: 11px;">Node markers every {NODE_INTERVAL_MILES} miles</p>
                <div style="background: #ecf0f1; padding: 8px; border-radius: 4px; margin: 10px 0; font-size: 11px;">
            <b>Map Color Legend:</b><br>
            <span style="color: #e74c3c;">━━━</span> Ascent (>2%)<br>
            <span style="color: #27ae60;">━━━</span> Descent (<-2%)<br>
            <span style="color: #3498db;">━━━</span> Flat (±2%)<br>
            <span style="color: #9b59b6;">━━━</span> Tunnel
        </div>
                <hr style="border: 1px solid #ecf0f1;">
        
        <h4 style="color: #3498db; margin-bottom: 8px;">📏 Distance & Time</h4>
        <table style="width:100%; font-size:12px; margin-bottom:15px;">
            <tr><td><b>Distance:</b></td><td>{analysis['distance_km']} km ({analysis['distance_miles']} mi)</td></tr>
            <tr><td><b>Duration:</b></td><td>{analysis['duration_min']} min</td></tr>
            <tr><td><b>Avg Speed:</b></td><td>{analysis['avg_speed_kmh']} km/h</td></tr>
        </table>
        
        <h4 style="color: #27ae60; margin-bottom: 8px;">⛰️ Elevation Profile</h4>
        <table style="width:100%; font-size:12px; margin-bottom:10px;">
            <tr><td><b>Ascent:</b></td><td style="color: #e74c3c;">{analysis['elevation']['ascent_m']} m ↑</td></tr>
            <tr><td><b>Descent:</b></td><td style="color: #3498db;">{analysis['elevation']['descent_m']} m ↓</td></tr>
            <tr><td><b>Net Elevation:</b></td><td>{analysis['elevation']['net_elevation_m']} m</td></tr>
        </table>
        
        <div style="background: linear-gradient(to right, #27ae60 0%, #e74c3c 50%, #27ae60 100%); 
                    height: 60px; border-radius: 4px; position: relative; margin-bottom: 15px;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">
                Elevation Change
            </div>
        </div>
        
        <h4 style="color: #e67e22; margin-bottom: 8px;">📐 Slope Analysis</h4>
        <table style="width:100%; font-size:12px; margin-bottom:10px;">
            <tr><td><b>Uphill segments:</b></td><td>{analysis['slopes']['uphill_segments']}</td></tr>
            <tr><td><b>Downhill segments:</b></td><td>{analysis['slopes']['downhill_segments']}</td></tr>
            <tr><td><b>Steep (>5%):</b></td><td style="color: #e74c3c;">{analysis['slopes']['steep_segments_over_5pct']}</td></tr>
            <tr><td><b>Max uphill:</b></td><td style="color: #e74c3c;">{analysis['slopes']['max_uphill_grade_pct']}%</td></tr>
            <tr><td><b>Max downhill:</b></td><td style="color: #3498db;">{analysis['slopes']['max_downhill_grade_pct']}%</td></tr>
        </table>
        
        <div style="background-color: #ecf0f1; padding: 8px; border-radius: 4px; margin-bottom: 15px;">
            <div style="background-color: #e74c3c; width: {min(100, analysis['slopes']['steep_segments_over_5pct']*3)}%; 
                        height: 20px; border-radius: 3px; text-align: center; color: white; line-height: 20px; font-size: 11px;">
                Steep: {analysis['slopes']['steep_segments_over_5pct']}
            </div>
        </div>
        
        <h4 style="color: #9b59b6; margin-bottom: 8px;">🛣️ Road Classes</h4>
        <table style="width:100%; font-size:11px; margin-bottom:15px;">
            {''.join([f"<tr><td>{k.replace('_', ' ').title()}:</td><td><b>{v}</b> segments</td></tr>" for k, v in road_classes.items()])}
        </table>
        
        <h4 style="color: #34495e; margin-bottom: 8px;">🏗️ Road Surfaces</h4>
        <table style="width:100%; font-size:11px; margin-bottom:15px;">
            {''.join([f"<tr><td>{k.replace('_', ' ').title()}:</td><td><b>{v}</b> segments</td></tr>" for k, v in surfaces.items()])}
        </table>
        
        <h4 style="color: #16a085; margin-bottom: 8px;">🌉 Environments</h4>
        <table style="width:100%; font-size:11px; margin-bottom:10px;">
            {''.join([f"<tr><td>{k.replace('_', ' ').title()}:</td><td><b>{v}</b> segments</td></tr>" for k, v in environments.items()])}
        </table>
        
        <hr style="border: 1px solid #ecf0f1;">
        <p style="font-size: 10px; color: #95a5a6; margin-bottom: 0;">
            Total waypoints: {analysis['num_points']}<br>
            Nodes for graph: {len(nodes)}
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))
    
    # Save map
    m.save(output_path)
    print(f"  ✓ Map saved to {output_path}")

def generate_routes(
    source: str,
    destination: str,
    profile: str = "truck_diesel",
    num_alternatives: int = 3,
    output_dir: str = "../Maps"
):
    """
    Main function to generate routes with alternatives, analysis, and maps.
    
    Args:
        source: Source location name or "lat,lon"
        destination: Destination location name or "lat,lon"
        profile: GraphHopper profile (default: truck_diesel)
        num_alternatives: Number of alternative routes to generate
        output_dir: Directory to save output files
    """
    print(f"\n{'='*80}")
    print(f"Route Generation: {source} → {destination}")
    print(f"Profile: {profile} | Alternatives: {num_alternatives}")
    print(f"{'='*80}\n")
    
    # Load location database
    locations_df = load_locations()
    
    # Parse source coordinates
    if ',' in source:
        start_coords = tuple(map(float, source.split(',')))
        start_name = f"Point ({start_coords[0]:.4f}, {start_coords[1]:.4f})"
    else:
        start_coords = get_location_coords(source, locations_df)
        if not start_coords:
            raise ValueError(f"Location not found: {source}")
        start_name = source
    
    # Parse destination coordinates
    if ',' in destination:
        end_coords = tuple(map(float, destination.split(',')))
        end_name = f"Point ({end_coords[0]:.4f}, {end_coords[1]:.4f})"
    else:
        end_coords = get_location_coords(destination, locations_df)
        if not end_coords:
            raise ValueError(f"Location not found: {destination}")
        end_name = destination
    
    print(f"Origin: {start_name} ({start_coords[0]:.4f}, {start_coords[1]:.4f})")
    print(f"Destination: {end_name} ({end_coords[0]:.4f}, {end_coords[1]:.4f})")
    print(f"\nQuerying GraphHopper for {num_alternatives} alternative routes...\n")
    
    # Query routes
    paths = query_route_alternatives(start_coords, end_coords, profile, num_alternatives)
    
    print(f"Found {len(paths)} alternative route(s)\n")
    
    # Process each route
    routes = []
    for idx, path in enumerate(paths):
        print(f"Alternative {idx + 1}:")
        
        # Analyze route
        analysis = analyze_route_details(path)
        print(f"  Distance: {analysis['distance_km']} km ({analysis['distance_miles']} mi)")
        print(f"  Duration: {analysis['duration_min']} min")
        print(f"  Elevation: +{analysis['elevation']['ascent_m']}m / -{analysis['elevation']['descent_m']}m")
        print(f"  Slopes: {analysis['slopes']['uphill_segments']} uphill, "
              f"{analysis['slopes']['downhill_segments']} downhill, "
              f"{analysis['slopes']['steep_segments_over_5pct']} steep (>5%)")
        
        # Create nodes every 10 miles
        print(f"  Creating nodes every {NODE_INTERVAL_MILES} miles...")
        points = path["points"]["coordinates"]
        nodes = create_route_nodes(points, NODE_INTERVAL_KM)
        print(f"  ✓ Created {len(nodes)} nodes")
        
        routes.append({
            "alternative_id": idx + 1,
            "path": path,
            "analysis": analysis,
            "nodes": nodes
        })
        print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = source.replace(' ', '_').replace(',', '')
    safe_dest = destination.replace(' ', '_').replace(',', '')
    base_name = f"{safe_source}_to_{safe_dest}_{timestamp}"
    
    # Save JSON with full route data
    json_path = os.path.join(output_dir, f"{base_name}.json")
    output_data = {
        "source": start_name,
        "destination": end_name,
        "source_coords": start_coords,
        "destination_coords": end_coords,
        "profile": profile,
        "timestamp": timestamp,
        "num_alternatives": len(routes),
        "routes": routes
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Route data saved to {json_path}")
    
    # Create separate HTML map for each alternative
    print(f"\nGenerating HTML maps for each alternative...")
    html_files = []
    for route in routes:
        alt_id = route['alternative_id']
        html_path = os.path.join(output_dir, f"{base_name}_alt{alt_id}.html")
        create_html_map(route, start_name, end_name, html_path)
        html_files.append(html_path)
    
    print(f"\n{'='*80}")
    print(f"Route generation complete!")
    print(f"  JSON: {json_path}")
    for idx, html_file in enumerate(html_files, 1):
        print(f"  Map Alt {idx}: {html_file}")
    print(f"{'='*80}\n")
    
    return output_data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_route.py <source> <destination> [profile] [num_alternatives]")
        print("\nExamples:")
        print("  python generate_route.py Antwerp Brussels")
        print("  python generate_route.py Ghent Liège truck_ev 5")
        print("  python generate_route.py 51.2194,4.4025 50.8503,4.3517")
        sys.exit(1)
    
    source = sys.argv[1]
    destination = sys.argv[2]
    profile = sys.argv[3] if len(sys.argv) > 3 else "truck_diesel"
    num_alternatives = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    try:
        generate_routes(source, destination, profile, num_alternatives)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
