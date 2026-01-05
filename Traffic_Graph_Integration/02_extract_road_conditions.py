"""
Step 2: Extract Road Conditions from Routes

Extracts detailed road features from all collected routes:
- Speed limits per segment
- Slopes/gradients  
- Road classes (motorway, primary, etc.)
- Surfaces
- Road environment
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
ROUTES_DIR = BASE_DIR / "routes"
FEATURES_DIR = BASE_DIR / "road_features"

FEATURES_DIR.mkdir(exist_ok=True)


def extract_features_from_route(route_data):
    """Extract all road features from a route"""
    
    path = route_data['paths'][0]
    coords = path['points']['coordinates']
    details = path.get('details', {})
    
    features = {
        'metadata': route_data['metadata'],
        'total_distance_km': path['distance'] / 1000,
        'total_time_min': path['time'] / 60000,
        'total_points': len(coords),
        'coordinates': coords,
        'features': {}
    }
    
    # Extract each feature type
    feature_types = ['max_speed', 'average_speed', 'road_class', 'surface', 
                     'road_environment', 'average_slope', 'max_slope']
    
    for feature_type in feature_types:
        if feature_type in details:
            feature_data = []
            
            for segment in details[feature_type]:
                feature_data.append({
                    'start_index': segment[0],
                    'end_index': segment[1],
                    'value': segment[2]
                })
            
            features['features'][feature_type] = feature_data
    
    return features


def main():
    print("=" * 80)
    print("Step 2: Road Conditions Extraction")
    print("=" * 80)
    
    route_files = sorted(ROUTES_DIR.glob("S*_to_D*.json"))
    
    if not route_files:
        print("❌ No route files found!")
        return
    
    print(f"\n📍 Processing {len(route_files)} routes\n")
    
    for route_file in route_files:
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        od_pair = route_data['metadata']['od_pair']
        
        # Extract features
        features = extract_features_from_route(route_data)
        
        # Save features
        output_file = FEATURES_DIR / f"{od_pair}_features.json"
        with open(output_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Print summary
        print(f"✓ {od_pair}:")
        print(f"  Distance: {features['total_distance_km']:.1f} km")
        print(f"  Points: {features['total_points']}")
        print(f"  Features: {', '.join(features['features'].keys())}")
        
        # Feature statistics
        if 'max_speed' in features['features']:
            speeds = [s['value'] for s in features['features']['max_speed'] if s['value'] is not None]
            if speeds:
                print(f"  Speed range: {min(speeds)}-{max(speeds)} km/h")
        
        if 'average_slope' in features['features']:
            slopes = [s['value'] for s in features['features']['average_slope'] if s['value'] is not None]
            if slopes:
                print(f"  Slope range: {min(slopes):.1f}% to {max(slopes):.1f}%")
        
        print()
    
    print("=" * 80)
    print(f"✓ Features saved to: {FEATURES_DIR}")
    print(f"  Files: {len(list(FEATURES_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
