#!/usr/bin/env python3
"""
Extract 4-6 midway points from selected routes.
Midpoints are evenly distributed along the route path.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple
import math

def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """Calculate distance between two points (lon, lat) in meters using Haversine formula."""
    lon1, lat1 = point1[0], point1[1]
    lon2, lat2 = point2[0], point2[1]
    
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_cumulative_distances(coordinates: List[List[float]]) -> List[float]:
    """Calculate cumulative distance along the route."""
    cumulative = [0.0]
    
    for i in range(1, len(coordinates)):
        dist = calculate_distance(coordinates[i-1], coordinates[i])
        cumulative.append(cumulative[-1] + dist)
    
    return cumulative

def extract_midway_points(route_data: dict, num_midpoints: int = 5) -> dict:
    """
    Extract evenly spaced midway points along the route.
    
    Args:
        route_data: Route JSON data
        num_midpoints: Number of midway points (4-6)
    
    Returns:
        Dictionary with start, midpoints, and end
    """
    route = route_data['routes'][0]  # Use alternative 1
    coordinates = route['path']['points']['coordinates']
    
    # Calculate cumulative distances
    cumulative_dist = calculate_cumulative_distances(coordinates)
    total_distance = cumulative_dist[-1]
    
    # Extract start and end points
    start_point = {
        'index': 0,
        'coordinates': coordinates[0],
        'lat': coordinates[0][1],
        'lon': coordinates[0][0],
        'distance_from_start_km': 0.0,
        'label': 'START'
    }
    
    end_point = {
        'index': len(coordinates) - 1,
        'coordinates': coordinates[-1],
        'lat': coordinates[-1][1],
        'lon': coordinates[-1][0],
        'distance_from_start_km': total_distance / 1000,
        'label': 'END'
    }
    
    # Calculate midway points (evenly distributed)
    midpoints = []
    segment_distance = total_distance / (num_midpoints + 1)
    
    for i in range(1, num_midpoints + 1):
        target_distance = segment_distance * i
        
        # Find closest point to target distance
        closest_idx = 0
        min_diff = abs(cumulative_dist[0] - target_distance)
        
        for idx, dist in enumerate(cumulative_dist):
            diff = abs(dist - target_distance)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx
        
        midpoint = {
            'index': closest_idx,
            'coordinates': coordinates[closest_idx],
            'lat': coordinates[closest_idx][1],
            'lon': coordinates[closest_idx][0],
            'distance_from_start_km': cumulative_dist[closest_idx] / 1000,
            'label': f'MIDPOINT_{i}'
        }
        midpoints.append(midpoint)
    
    result = {
        'source': route_data['source'],
        'destination': route_data['destination'],
        'total_distance_km': total_distance / 1000,
        'total_points': len(coordinates),
        'start': start_point,
        'midpoints': midpoints,
        'end': end_point,
        'all_waypoints': [start_point] + midpoints + [end_point]
    }
    
    return result

def main():
    selected_routes_dir = "../selected_routes"
    output_dir = "../midway_points"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MIDWAY POINT EXTRACTION")
    print("="*80)
    
    # Get selected route files
    route_files = list(Path(selected_routes_dir).glob('*.json'))
    route_files = [f for f in route_files if f.name != 'selection_summary.json']
    
    print(f"\nProcessing {len(route_files)} routes...")
    
    all_midway_data = []
    
    for route_file in route_files:
        print(f"\n{'='*80}")
        print(f"Route: {route_file.stem}")
        print(f"{'='*80}")
        
        # Load route
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        # Extract midway points
        midway_data = extract_midway_points(route_data, num_midpoints=5)
        
        # Display info
        print(f"\nRoute: {midway_data['source']} → {midway_data['destination']}")
        print(f"Total Distance: {midway_data['total_distance_km']:.1f} km")
        print(f"Total Points: {midway_data['total_points']}")
        print(f"\nWaypoints:")
        
        for waypoint in midway_data['all_waypoints']:
            print(f"  {waypoint['label']:12s} @ {waypoint['distance_from_start_km']:6.1f} km "
                  f"| Lat: {waypoint['lat']:.6f}, Lon: {waypoint['lon']:.6f}")
        
        # Save to file
        output_file = Path(output_dir) / f"{route_file.stem}_midpoints.json"
        with open(output_file, 'w') as f:
            json.dump(midway_data, f, indent=2)
        
        print(f"\n✓ Saved to {output_file.name}")
        
        all_midway_data.append(midway_data)
    
    # Save combined summary
    summary_file = Path(output_dir) / "all_midpoints_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_midway_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ All midpoint data saved to {summary_file}")
    print(f"{'='*80}\n")
    
    # Summary statistics
    print("SUMMARY")
    print("="*80)
    total_waypoints = sum(len(data['all_waypoints']) for data in all_midway_data)
    print(f"Total routes: {len(all_midway_data)}")
    print(f"Total waypoints: {total_waypoints}")
    print(f"Estimated API calls: {total_waypoints} × 24h × 7days = {total_waypoints * 24 * 7:,}")
    print("="*80)

if __name__ == "__main__":
    main()
