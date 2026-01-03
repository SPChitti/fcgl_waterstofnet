#!/usr/bin/env python3
"""
Extract route features and speed limits from the selected routes.
Creates organized files with road characteristics for analysis.
"""

import json
from pathlib import Path
from typing import Dict, List

def extract_route_features(route_file: Path) -> Dict:
    """
    Extract all route features from a route JSON file.
    """
    with open(route_file, 'r') as f:
        data = json.load(f)
    
    route = data['routes'][0]  # Use alternative 1
    path = route['path']
    details = path.get('details', {})
    
    # Get total distance and time
    total_distance_m = path['distance']
    total_time_ms = path['time']
    num_points = len(path['points']['coordinates'])
    
    features = {
        'route_info': {
            'source': data['source'],
            'destination': data['destination'],
            'total_distance_km': total_distance_m / 1000,
            'total_time_min': total_time_ms / 60000,
            'total_points': num_points
        },
        'max_speed': [],
        'average_slope': [],
        'max_slope': [],
        'surface': [],
        'road_class': [],
        'road_environment': [],
        'road_access': []
    }
    
    # Extract each feature type
    for feature_type in ['max_speed', 'average_slope', 'max_slope', 
                         'surface', 'road_class', 'road_environment', 'road_access']:
        if feature_type in details:
            for segment in details[feature_type]:
                start_idx, end_idx, value = segment[0], segment[1], segment[2]
                
                features[feature_type].append({
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'value': value,
                    'num_points': end_idx - start_idx
                })
    
    return features

def create_summary_report(features: Dict) -> str:
    """Create human-readable summary report."""
    report = []
    info = features['route_info']
    
    report.append("="*80)
    report.append(f"ROUTE FEATURES: {info['source']} → {info['destination']}")
    report.append("="*80)
    report.append(f"Distance: {info['total_distance_km']:.1f} km")
    report.append(f"Time: {info['total_time_min']:.1f} minutes")
    report.append(f"Total Points: {info['total_points']}")
    report.append("")
    
    # Speed Limits
    if features['max_speed']:
        report.append("SPEED LIMITS:")
        report.append("-" * 80)
        speed_summary = {}
        for seg in features['max_speed']:
            speed = seg['value']
            if speed is not None:  # Skip None values
                if speed not in speed_summary:
                    speed_summary[speed] = 0
                speed_summary[speed] += seg['num_points']
        
        for speed in sorted([s for s in speed_summary.keys() if s is not None], reverse=True):
            points = speed_summary[speed]
            percentage = (points / info['total_points']) * 100
            # Handle both int and float speed values
            speed_str = f"{speed:.0f}" if isinstance(speed, float) else str(speed)
            report.append(f"  {speed_str:>5s} km/h: {points:4d} points ({percentage:5.1f}%)")
        report.append("")
    
    # Road Classes
    if features['road_class']:
        report.append("ROAD CLASSES:")
        report.append("-" * 80)
        class_names = {
            'motorway': 'Motorway',
            'trunk': 'Trunk Road',
            'primary': 'Primary Road',
            'secondary': 'Secondary Road',
            'tertiary': 'Tertiary Road',
            'unclassified': 'Unclassified',
            'residential': 'Residential',
            'service': 'Service Road'
        }
        class_summary = {}
        for seg in features['road_class']:
            road_class = seg['value']
            if road_class not in class_summary:
                class_summary[road_class] = 0
            class_summary[road_class] += seg['num_points']
        
        for road_class, points in sorted(class_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (points / info['total_points']) * 100
            name = class_names.get(road_class, road_class)
            report.append(f"  {name:20s}: {points:4d} points ({percentage:5.1f}%)")
        report.append("")
    
    # Slopes
    if features['average_slope']:
        report.append("SLOPE STATISTICS:")
        report.append("-" * 80)
        slopes = [seg['value'] for seg in features['average_slope']]
        avg_slope = sum(slopes) / len(slopes)
        max_slope = max([seg['value'] for seg in features['max_slope']]) if features['max_slope'] else 0
        min_slope = min([seg['value'] for seg in features['max_slope']]) if features['max_slope'] else 0
        
        report.append(f"  Average Slope: {avg_slope:.2f}%")
        report.append(f"  Max Uphill: {max_slope:.2f}%")
        report.append(f"  Max Downhill: {min_slope:.2f}%")
        report.append("")
    
    # Environment
    if features['road_environment']:
        report.append("ROAD ENVIRONMENT:")
        report.append("-" * 80)
        env_summary = {}
        for seg in features['road_environment']:
            env = seg['value']
            if env not in env_summary:
                env_summary[env] = 0
            env_summary[env] += seg['num_points']
        
        for env, points in sorted(env_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (points / info['total_points']) * 100
            report.append(f"  {env:15s}: {points:4d} points ({percentage:5.1f}%)")
        report.append("")
    
    # Surface
    if features['surface']:
        report.append("ROAD SURFACE:")
        report.append("-" * 80)
        surface_summary = {}
        for seg in features['surface']:
            surface = seg['value']
            if surface not in surface_summary:
                surface_summary[surface] = 0
            surface_summary[surface] += seg['num_points']
        
        for surface, points in sorted(surface_summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (points / info['total_points']) * 100
            report.append(f"  {surface:15s}: {points:4d} points ({percentage:5.1f}%)")
        report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)

def main():
    selected_routes_dir = "../selected_routes"
    output_dir = "../route_features"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*80)
    print("EXTRACTING ROUTE FEATURES AND SPEED LIMITS")
    print("="*80)
    
    # Get route files
    route_files = list(Path(selected_routes_dir).glob('*.json'))
    route_files = [f for f in route_files if f.name != 'selection_summary.json']
    
    print(f"\nProcessing {len(route_files)} routes...\n")
    
    for route_file in route_files:
        print(f"Processing: {route_file.stem}")
        
        # Extract features
        features = extract_route_features(route_file)
        
        # Save full features JSON
        output_json = Path(output_dir) / f"{route_file.stem}_features.json"
        with open(output_json, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"  ✓ Saved features to: {output_json.name}")
        
        # Create summary report
        report = create_summary_report(features)
        output_txt = Path(output_dir) / f"{route_file.stem}_summary.txt"
        with open(output_txt, 'w') as f:
            f.write(report)
        print(f"  ✓ Saved summary to: {output_txt.name}")
        
        # Print summary to console
        print("\n" + report + "\n")
    
    print("="*80)
    print("✓ All route features extracted successfully!")
    print(f"  Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
