"""
Step 5: Analyze Traffic Patterns and Map to Route Segments

Analyzes collected traffic data and maps it to route segments based on spatial proximity.
Calculates traffic multipliers for peak vs off-peak periods.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
ROUTES_DIR = BASE_DIR / "routes"
FEATURES_DIR = BASE_DIR / "road_features"
TRAFFIC_DIR = BASE_DIR / "traffic_data"
OUTPUT_DIR = BASE_DIR / "traffic_mapped"

OUTPUT_DIR.mkdir(exist_ok=True)


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


def load_waypoints():
    """Load waypoint definitions"""
    waypoints_file = TRAFFIC_DIR / "waypoints.json"
    with open(waypoints_file, 'r') as f:
        return json.load(f)


def load_traffic_data():
    """Load all traffic measurements"""
    
    traffic_by_waypoint = defaultdict(lambda: {'morning_peak': [], 'off_peak': []})
    
    # Scan all date directories
    date_dirs = [d for d in TRAFFIC_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for date_dir in date_dirs:
        # Morning peak
        morning_dir = date_dir / "morning_peak"
        if morning_dir.exists():
            for file in morning_dir.glob("waypoint_*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    waypoint_id = data['metadata']['waypoint_id']
                    
                    # Extract traffic metrics
                    flow_data = data.get('flowSegmentData', {})
                    traffic_by_waypoint[waypoint_id]['morning_peak'].append({
                        'currentSpeed': flow_data.get('currentSpeed'),
                        'freeFlowSpeed': flow_data.get('freeFlowSpeed'),
                        'currentTravelTime': flow_data.get('currentTravelTime'),
                        'freeFlowTravelTime': flow_data.get('freeFlowTravelTime'),
                        'confidence': flow_data.get('confidence'),
                        'date': data['metadata']['date']
                    })
        
        # Off peak
        off_peak_dir = date_dir / "off_peak"
        if off_peak_dir.exists():
            for file in off_peak_dir.glob("waypoint_*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    waypoint_id = data['metadata']['waypoint_id']
                    
                    flow_data = data.get('flowSegmentData', {})
                    traffic_by_waypoint[waypoint_id]['off_peak'].append({
                        'currentSpeed': flow_data.get('currentSpeed'),
                        'freeFlowSpeed': flow_data.get('freeFlowSpeed'),
                        'currentTravelTime': flow_data.get('currentTravelTime'),
                        'freeFlowTravelTime': flow_data.get('freeFlowTravelTime'),
                        'confidence': flow_data.get('confidence'),
                        'date': data['metadata']['date']
                    })
    
    return traffic_by_waypoint


def analyze_traffic_patterns(traffic_by_waypoint):
    """Calculate average traffic metrics per waypoint"""
    
    analysis = {}
    
    for waypoint_id, periods in traffic_by_waypoint.items():
        waypoint_analysis = {}
        
        for period_name, measurements in periods.items():
            if not measurements:
                continue
            
            # Calculate averages
            speeds = [m['currentSpeed'] for m in measurements if m['currentSpeed'] is not None]
            freeflow = [m['freeFlowSpeed'] for m in measurements if m['freeFlowSpeed'] is not None]
            travel_times = [m['currentTravelTime'] for m in measurements if m['currentTravelTime'] is not None]
            freeflow_times = [m['freeFlowTravelTime'] for m in measurements if m['freeFlowTravelTime'] is not None]
            
            if speeds and freeflow:
                avg_speed = np.mean(speeds)
                avg_freeflow = np.mean(freeflow)
                speed_ratio = avg_speed / avg_freeflow if avg_freeflow > 0 else 1.0
                
                waypoint_analysis[period_name] = {
                    'avg_current_speed': avg_speed,
                    'avg_freeflow_speed': avg_freeflow,
                    'speed_ratio': speed_ratio,
                    'congestion_factor': 1.0 - speed_ratio,
                    'avg_travel_time': np.mean(travel_times) if travel_times else None,
                    'avg_freeflow_time': np.mean(freeflow_times) if freeflow_times else None,
                    'num_measurements': len(measurements)
                }
        
        if waypoint_analysis:
            analysis[waypoint_id] = waypoint_analysis
    
    return analysis


def map_traffic_to_routes(waypoints, traffic_analysis):
    """Map traffic data to route segments using spatial proximity"""
    
    route_files = sorted(ROUTES_DIR.glob("S*_to_D*.json"))
    
    routes_with_traffic = {}
    
    for route_file in route_files:
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        od_pair = route_data['metadata']['od_pair']
        coords = route_data['paths'][0]['points']['coordinates']
        
        # For each route coordinate, find nearest waypoint with traffic data
        traffic_mapped = []
        
        for idx, (lon, lat) in enumerate(coords):
            # Find nearest waypoint
            min_dist = float('inf')
            nearest_waypoint = None
            
            for wp_idx, waypoint in enumerate(waypoints):
                dist = haversine_distance(lat, lon, waypoint['lat'], waypoint['lon'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_waypoint = wp_idx
            
            # Get traffic data for nearest waypoint
            traffic_data = None
            if nearest_waypoint is not None and nearest_waypoint in traffic_analysis:
                traffic_data = traffic_analysis[nearest_waypoint]
            
            traffic_mapped.append({
                'coord_index': idx,
                'lat': lat,
                'lon': lon,
                'nearest_waypoint': nearest_waypoint,
                'distance_to_waypoint': min_dist,
                'traffic': traffic_data
            })
        
        routes_with_traffic[od_pair] = {
            'metadata': route_data['metadata'],
            'coordinates': coords,
            'traffic_mapping': traffic_mapped
        }
    
    return routes_with_traffic


def calculate_statistics(routes_with_traffic):
    """Calculate overall statistics"""
    
    print("\n" + "=" * 80)
    print("TRAFFIC ANALYSIS SUMMARY")
    print("=" * 80)
    
    for od_pair, route_data in routes_with_traffic.items():
        print(f"\n{od_pair}:")
        
        # Count segments with traffic data
        segments_with_traffic = sum(1 for t in route_data['traffic_mapping'] if t['traffic'] is not None)
        total_segments = len(route_data['traffic_mapping'])
        
        print(f"  Coverage: {segments_with_traffic}/{total_segments} segments ({100*segments_with_traffic/total_segments:.1f}%)")
        
        # Average congestion
        morning_congestion = []
        offpeak_congestion = []
        
        for segment in route_data['traffic_mapping']:
            if segment['traffic']:
                if 'morning_peak' in segment['traffic']:
                    morning_congestion.append(segment['traffic']['morning_peak']['congestion_factor'])
                if 'off_peak' in segment['traffic']:
                    offpeak_congestion.append(segment['traffic']['off_peak']['congestion_factor'])
        
        if morning_congestion:
            print(f"  Morning peak congestion: {np.mean(morning_congestion):.2%} (avg)")
            print(f"    Speed ratio: {np.mean([1-c for c in morning_congestion]):.2f}")
        
        if offpeak_congestion:
            print(f"  Off-peak congestion: {np.mean(offpeak_congestion):.2%} (avg)")
            print(f"    Speed ratio: {np.mean([1-c for c in offpeak_congestion]):.2f}")


def main():
    print("=" * 80)
    print("Step 5: Traffic Pattern Analysis and Mapping")
    print("=" * 80)
    print()
    
    # Load waypoints
    print("Loading waypoints...")
    waypoints = load_waypoints()
    print(f"  ✓ {len(waypoints)} waypoints")
    
    # Load traffic data
    print("\nLoading traffic measurements...")
    traffic_by_waypoint = load_traffic_data()
    print(f"  ✓ {len(traffic_by_waypoint)} waypoints with traffic data")
    
    # Analyze patterns
    print("\nAnalyzing traffic patterns...")
    traffic_analysis = analyze_traffic_patterns(traffic_by_waypoint)
    print(f"  ✓ {len(traffic_analysis)} waypoints analyzed")
    
    # Map to routes
    print("\nMapping traffic to route segments...")
    routes_with_traffic = map_traffic_to_routes(waypoints, traffic_analysis)
    print(f"  ✓ {len(routes_with_traffic)} routes processed")
    
    # Save results
    print("\nSaving traffic-mapped routes...")
    for od_pair, route_data in routes_with_traffic.items():
        output_file = OUTPUT_DIR / f"{od_pair}_traffic.json"
        with open(output_file, 'w') as f:
            json.dump(route_data, f, indent=2)
        print(f"  ✓ {od_pair}")
    
    # Calculate statistics
    calculate_statistics(routes_with_traffic)
    
    print("\n" + "=" * 80)
    print(f"✓ Traffic-mapped routes saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
