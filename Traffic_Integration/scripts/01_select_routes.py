#!/usr/bin/env python3
"""
Select 3 representative routes from the 306 existing routes.
Criteria: Geographic diversity, distance variety, urban/rural mix
"""

import json
import os
from pathlib import Path

def analyze_route(route_file):
    """Extract key characteristics of a route."""
    with open(route_file, 'r') as f:
        data = json.load(f)
    
    route = data['routes'][0]  # Use alternative 1
    distance_km = route['path']['distance'] / 1000
    time_min = route['path']['time'] / 60000
    num_points = len(route['path']['points']['coordinates'])
    
    return {
        'file': route_file.name,
        'source': data['source'],
        'destination': data['destination'],
        'distance_km': distance_km,
        'time_min': time_min,
        'num_points': num_points,
        'od_pair': f"{data['source']} → {data['destination']}"
    }

def select_representative_routes(routes_dir, num_routes=3):
    """
    Select representative routes based on:
    - Short route (~20-40km)
    - Medium route (~50-80km)
    - Long route (>80km)
    - Geographic diversity
    """
    routes_dir = Path(routes_dir)
    json_files = [f for f in routes_dir.glob('*.json') 
                  if not f.name.startswith('batch_summary') 
                  and not f.name.startswith('failed_pairs')]
    
    print(f"Analyzing {len(json_files)} routes...")
    
    # Analyze all routes
    routes_info = []
    for route_file in json_files:
        try:
            info = analyze_route(route_file)
            routes_info.append(info)
        except Exception as e:
            print(f"Error analyzing {route_file.name}: {e}")
            continue
    
    # Sort by distance
    routes_info.sort(key=lambda x: x['distance_km'])
    
    # Select candidates
    # Short route: 20-40km range
    short_candidates = [r for r in routes_info if 20 <= r['distance_km'] <= 40]
    # Medium route: 50-80km range  
    medium_candidates = [r for r in routes_info if 50 <= r['distance_km'] <= 80]
    # Long route: >80km
    long_candidates = [r for r in routes_info if r['distance_km'] > 80]
    
    print(f"\nRoute categories:")
    print(f"  Short (20-40km): {len(short_candidates)} routes")
    print(f"  Medium (50-80km): {len(medium_candidates)} routes")
    print(f"  Long (>80km): {len(long_candidates)} routes")
    
    # Select one from each category
    selected = []
    
    # Short route - pick middle of range for stability
    if short_candidates:
        selected.append(short_candidates[len(short_candidates)//2])
    
    # Medium route - pick middle of range
    if medium_candidates:
        selected.append(medium_candidates[len(medium_candidates)//2])
    
    # Long route - pick middle of range
    if long_candidates:
        selected.append(long_candidates[len(long_candidates)//2])
    
    return selected

def main():
    routes_dir = "../../Maps/batch_routes"
    output_dir = "../selected_routes"
    
    print("="*80)
    print("ROUTE SELECTION")
    print("="*80)
    
    # Select routes
    selected = select_representative_routes(routes_dir, num_routes=3)
    
    print("\n" + "="*80)
    print("SELECTED ROUTES")
    print("="*80)
    
    for i, route in enumerate(selected, 1):
        print(f"\nRoute {i}: {route['od_pair']}")
        print(f"  Distance: {route['distance_km']:.1f} km")
        print(f"  Time: {route['time_min']:.1f} min")
        print(f"  Points: {route['num_points']}")
        print(f"  File: {route['file']}")
    
    # Copy selected routes to output directory
    print("\n" + "="*80)
    print("COPYING ROUTES")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for route in selected:
        src = Path(routes_dir) / route['file']
        dst = Path(output_dir) / route['file']
        
        with open(src, 'r') as f:
            data = json.load(f)
        
        with open(dst, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Copied {route['file']}")
    
    # Save selection summary
    summary_file = Path(output_dir) / "selection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(selected, f, indent=2)
    
    print(f"\n✓ Selection summary saved to {summary_file}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
