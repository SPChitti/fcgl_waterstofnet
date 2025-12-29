#!/usr/bin/env python3
"""
Extract route paths as sequences of (lat, lon) coordinates.
Generates one path per route for graph creation.
"""

import json
import os
import sys
from typing import List, Dict

def extract_route_paths(
    routes_dir: str = "../Maps/batch_routes",
    output_file: str = "../Data/route_paths.json",
    alternative: int = 1
):
    """
    Extract path coordinates from all route files.
    
    Args:
        routes_dir: Directory containing route JSON files
        output_file: Output JSON file path
        alternative: Which alternative route to use (1, 2, or 3)
    """
    print("="*80)
    print(f"Extracting Route Paths for Graph Creation")
    print(f"Alternative: {alternative}")
    print("="*80 + "\n")
    
    # Get all route JSON files
    json_files = sorted([
        f for f in os.listdir(routes_dir) 
        if f.endswith('.json') 
        and not f.startswith('batch_summary') 
        and not f.startswith('failed_pairs')
    ])
    
    print(f"Found {len(json_files)} route files\n")
    
    route_paths = []
    
    for idx, filename in enumerate(json_files, 1):
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(json_files)}...")
        
        filepath = os.path.join(routes_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Get the specified alternative
            routes = data.get('routes', [])
            if alternative > len(routes):
                print(f"  ⚠️  Skipping {filename}: Only {len(routes)} alternatives available")
                continue
            
            route = routes[alternative - 1]
            nodes = route.get('nodes', [])
            
            if not nodes:
                print(f"  ⚠️  Skipping {filename}: No nodes found")
                continue
            
            # Extract path as list of [lat, lon]
            path = [[node['latitude'], node['longitude']] for node in nodes]
            
            # Create route path entry
            route_path = {
                "route_id": idx,
                "origin": data['source'],
                "destination": data['destination'],
                "origin_coords": list(data['source_coords']),
                "destination_coords": list(data['destination_coords']),
                "alternative_id": alternative,
                "num_nodes": len(path),
                "path": path
            }
            
            route_paths.append(route_path)
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        "metadata": {
            "total_routes": len(route_paths),
            "alternative_used": alternative,
            "description": "Route paths with node coordinates for graph creation"
        },
        "routes": route_paths
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Extraction complete!")
    print(f"  Total routes: {len(route_paths)}")
    print(f"  Output file: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("="*80 + "\n")
    
    # Show sample
    if route_paths:
        print("Sample route:")
        sample = route_paths[0]
        print(f"  Route {sample['route_id']}: {sample['origin']} → {sample['destination']}")
        print(f"  Nodes: {sample['num_nodes']}")
        print(f"  Path: {sample['path'][:3]} ... {sample['path'][-1:]}")
        print()
    
    return output_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract route paths for graph creation")
    parser.add_argument("--routes-dir", default="../Maps/batch_routes",
                       help="Directory containing route JSON files")
    parser.add_argument("--output", default="../Data/route_paths.json",
                       help="Output JSON file path")
    parser.add_argument("--alternative", type=int, default=1,
                       help="Which alternative route to use (1, 2, or 3)")
    
    args = parser.parse_args()
    
    extract_route_paths(
        routes_dir=args.routes_dir,
        output_file=args.output,
        alternative=args.alternative
    )
