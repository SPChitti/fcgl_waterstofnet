"""
Step 1: Query GraphHopper for All Source-Destination Routes

Queries GraphHopper for multiple alternative routes between all source-destination pairs:
- S1 (Genk) → D1, D2, D3
- S2 (Antwerp) → D1, D2, D3

Total: 6 OD pairs × 10 alternatives = 60 routes
"""

import requests
import json
import yaml
import time
from pathlib import Path
from datetime import datetime

# Directories
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.yaml"
ROUTES_DIR = BASE_DIR / "routes"

ROUTES_DIR.mkdir(exist_ok=True)


def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)


def query_graphhopper(origin_lat, origin_lon, dest_lat, dest_lon, num_alternatives=10):
    """
    Query GraphHopper for multiple alternative routes
    
    Args:
        origin_lat, origin_lon: Starting point coordinates
        dest_lat, dest_lon: Destination coordinates
        num_alternatives: Number of alternative routes to request
        
    Returns:
        dict: GraphHopper response with routes
    """
    url = "http://localhost:8080/route"
    
    params = {
        'point': [f"{origin_lat},{origin_lon}", f"{dest_lat},{dest_lon}"],
        'profile': 'truck_diesel',
        'points_encoded': 'false',
        'details': [
            'max_speed',
            'average_speed', 
            'road_class',
            'surface',
            'road_environment',
            'average_slope',
            'max_slope'
        ],
        'alternative_route.max_paths': num_alternatives,
        'alternative_route.max_weight_factor': 1.5,
        'alternative_route.max_share_factor': 0.6
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error: {str(e)}")
        return None


def save_routes(od_pair, routes_data):
    """Save routes to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{od_pair}_{timestamp}.json"
    filepath = ROUTES_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(routes_data, f, indent=2)
    
    return filepath


def main():
    print("=" * 80)
    print("Step 1: GraphHopper Route Collection")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    sources = config['locations']['sources']
    destinations = config['locations']['destinations']
    num_alternatives = config['graphhopper']['alternatives']
    
    print(f"\n📍 Sources: {len(sources)}")
    for s_id, s_data in sources.items():
        print(f"   {s_id}: {s_data['name']} ({s_data['lat']}, {s_data['lon']})")
    
    print(f"\n📍 Destinations: {len(destinations)}")
    for d_id, d_data in destinations.items():
        print(f"   {d_id}: {d_data['name']} ({d_data['lat']}, {d_data['lon']})")
    
    total_pairs = len(sources) * len(destinations)
    print(f"\n🔄 Total OD pairs: {total_pairs}")
    print(f"🔄 Alternatives per pair: {num_alternatives}")
    print(f"🔄 Expected total routes: {total_pairs * num_alternatives}")
    
    # Collect routes
    print("\n" + "=" * 80)
    print("Starting route collection...")
    print("=" * 80)
    
    collected = 0
    failed = 0
    
    for s_id, s_data in sources.items():
        for d_id, d_data in destinations.items():
            od_pair = f"{s_id}_to_{d_id}"
            
            print(f"\n📍 {od_pair}: {s_data['name']} → {d_data['name']}")
            
            # Query GraphHopper
            routes_data = query_graphhopper(
                s_data['lat'], s_data['lon'],
                d_data['lat'], d_data['lon'],
                num_alternatives
            )
            
            if routes_data and 'paths' in routes_data:
                num_routes = len(routes_data['paths'])
                
                # Add metadata
                routes_data['metadata'] = {
                    'od_pair': od_pair,
                    'source_id': s_id,
                    'source_name': s_data['name'],
                    'source_coords': [s_data['lat'], s_data['lon']],
                    'destination_id': d_id,
                    'destination_name': d_data['name'],
                    'destination_coords': [d_data['lat'], d_data['lon']],
                    'num_alternatives': num_routes,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to file
                filepath = save_routes(od_pair, routes_data)
                
                # Print summary
                total_dist = sum(p['distance'] for p in routes_data['paths']) / 1000
                avg_dist = total_dist / num_routes
                
                print(f"  ✓ Found {num_routes} routes")
                print(f"  ✓ Avg distance: {avg_dist:.1f} km")
                print(f"  ✓ Saved: {filepath.name}")
                
                collected += num_routes
                
            else:
                print(f"  ✗ Failed to get routes")
                failed += 1
            
            # Small delay to be nice to GraphHopper
            time.sleep(0.5)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Collection Complete!")
    print("=" * 80)
    
    print(f"\n📊 Statistics:")
    print(f"   OD pairs processed: {total_pairs}")
    print(f"   Routes collected: {collected}")
    print(f"   Failed queries: {failed}")
    print(f"   Average routes per pair: {collected / total_pairs:.1f}")
    
    print(f"\n✓ Routes saved to: {ROUTES_DIR}")
    print(f"   Files: {len(list(ROUTES_DIR.glob('*.json')))}")
    
    # Create summary
    summary = {
        'collection_date': datetime.now().isoformat(),
        'num_sources': len(sources),
        'num_destinations': len(destinations),
        'num_od_pairs': total_pairs,
        'routes_collected': collected,
        'failed_queries': failed,
        'alternatives_requested': num_alternatives
    }
    
    summary_file = ROUTES_DIR / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary: {summary_file.name}")


if __name__ == "__main__":
    main()
