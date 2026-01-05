"""
Step 4: Collect Traffic Data for Routes

Samples waypoints across all 6 routes and collects TomTom traffic data for:
- Morning peak (7-9 AM)
- Off-peak (2-4 AM)
- Multiple days for pattern analysis
"""

import json
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).parent
ROUTES_DIR = BASE_DIR / "routes"
TRAFFIC_DIR = BASE_DIR / "traffic_data"
CONFIG_FILE = BASE_DIR / "config.yaml"

TRAFFIC_DIR.mkdir(exist_ok=True)

# TomTom API configuration
TOMTOM_API_KEY = "LgYlpY5lyNXdQuFI8hbumVMx1KmXbhZA"
TRAFFIC_API_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"

# Time periods to collect
TIME_PERIODS = {
    'morning_peak': 8,  # 8 AM
    'off_peak': 3       # 3 AM
}


def load_routes():
    """Load all route coordinates"""
    routes = {}
    
    route_files = sorted(ROUTES_DIR.glob("S*_to_D*.json"))
    
    for route_file in route_files:
        with open(route_file, 'r') as f:
            data = json.load(f)
        
        od_pair = data['metadata']['od_pair']
        coords = data['paths'][0]['points']['coordinates']
        
        routes[od_pair] = {
            'coords': coords,
            'metadata': data['metadata']
        }
    
    return routes


def sample_waypoints(routes, target_count=100):
    """
    Sample waypoints across all routes
    
    Strategy:
    - Sample proportional to route length
    - Ensure good coverage of both unique and shared sections
    """
    
    waypoints = []
    
    # Calculate total points across all routes
    total_points = sum(len(r['coords']) for r in routes.values())
    
    for od_pair, route_data in routes.items():
        coords = route_data['coords']
        route_length = len(coords)
        
        # Number of waypoints for this route (proportional to length)
        route_waypoints = int(target_count * route_length / total_points)
        route_waypoints = max(10, route_waypoints)  # At least 10 per route
        
        # Sample evenly along the route
        step = max(1, route_length // route_waypoints)
        
        for i in range(0, route_length, step):
            lon, lat = coords[i]
            waypoints.append({
                'lat': lat,
                'lon': lon,
                'route': od_pair,
                'index': i
            })
    
    print(f"Sampled {len(waypoints)} waypoints across {len(routes)} routes")
    
    # Save waypoints
    waypoints_file = TRAFFIC_DIR / "waypoints.json"
    with open(waypoints_file, 'w') as f:
        json.dump(waypoints, f, indent=2)
    
    return waypoints


def query_traffic(lat, lon, hour):
    """Query TomTom Traffic API for a specific location and time"""
    
    params = {
        'key': TOMTOM_API_KEY,
        'point': f"{lat},{lon}",
        'unit': 'KMPH'
    }
    
    try:
        response = requests.get(TRAFFIC_API_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"    ⚠ API error {response.status_code}: {response.text[:100]}")
            return None
            
    except Exception as e:
        print(f"    ⚠ Request failed: {e}")
        return None


def collect_traffic_data(waypoints, days=7):
    """
    Collect traffic data for all waypoints across time periods and days
    
    Args:
        waypoints: List of waypoint dicts with lat/lon
        days: Number of days to collect (recent historical data)
    """
    
    print("\n" + "=" * 80)
    print("TRAFFIC DATA COLLECTION")
    print("=" * 80)
    print(f"Waypoints: {len(waypoints)}")
    print(f"Time periods: {len(TIME_PERIODS)} (morning peak, off-peak)")
    print(f"Days: {days}")
    print(f"Total API calls: {len(waypoints) * len(TIME_PERIODS) * days}")
    print("=" * 80)
    print()
    
    stats = {
        'total_calls': 0,
        'successful': 0,
        'failed': 0,
        'start_time': datetime.now()
    }
    
    # Collect for each day
    for day in range(days):
        date = datetime.now() - timedelta(days=day)
        date_str = date.strftime('%Y%m%d')
        
        print(f"\n📅 Day {day+1}/{days}: {date.strftime('%A, %B %d, %Y')}")
        
        # For each time period
        for period_name, hour in TIME_PERIODS.items():
            print(f"\n  ⏰ {period_name.replace('_', ' ').title()} ({hour}:00)")
            
            period_dir = TRAFFIC_DIR / date_str / period_name
            period_dir.mkdir(parents=True, exist_ok=True)
            
            # Query each waypoint
            for idx, waypoint in enumerate(waypoints):
                stats['total_calls'] += 1
                
                if idx % 20 == 0:
                    print(f"    Progress: {idx}/{len(waypoints)} waypoints...")
                
                # Query traffic
                traffic_data = query_traffic(waypoint['lat'], waypoint['lon'], hour)
                
                if traffic_data:
                    stats['successful'] += 1
                    
                    # Save response
                    filename = f"waypoint_{idx:04d}_{waypoint['route']}.json"
                    filepath = period_dir / filename
                    
                    # Add metadata
                    traffic_data['metadata'] = {
                        'waypoint_id': idx,
                        'route': waypoint['route'],
                        'waypoint_index': waypoint['index'],
                        'query_lat': waypoint['lat'],
                        'query_lon': waypoint['lon'],
                        'date': date_str,
                        'period': period_name,
                        'hour': hour,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(traffic_data, f, indent=2)
                else:
                    stats['failed'] += 1
                
                # Rate limiting: small delay between requests
                time.sleep(0.1)
            
            print(f"    ✓ Completed {len(waypoints)} waypoints")
    
    # Final statistics
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total API calls: {stats['total_calls']}")
    print(f"Successful: {stats['successful']} ({100*stats['successful']/stats['total_calls']:.1f}%)")
    print(f"Failed: {stats['failed']} ({100*stats['failed']/stats['total_calls']:.1f}%)")
    print(f"Duration: {stats['duration']/60:.1f} minutes")
    print(f"Average: {stats['duration']/stats['total_calls']:.2f} seconds per call")
    print("=" * 80)
    
    # Save statistics
    with open(TRAFFIC_DIR / "collection_stats.json", 'w') as f:
        json.dump({
            'total_calls': stats['total_calls'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'duration_seconds': stats['duration'],
            'start_time': stats['start_time'].isoformat(),
            'end_time': stats['end_time'].isoformat()
        }, f, indent=2)


def main():
    print("=" * 80)
    print("Step 4: Traffic Data Collection")
    print("=" * 80)
    print()
    
    # Load routes
    routes = load_routes()
    print(f"✓ Loaded {len(routes)} routes\n")
    
    # Sample waypoints
    waypoints = sample_waypoints(routes, target_count=102)
    
    # Confirm before proceeding
    print("\n" + "=" * 80)
    print("READY TO COLLECT TRAFFIC DATA")
    print("=" * 80)
    print(f"This will make approximately {len(waypoints) * len(TIME_PERIODS) * 7} API calls")
    print("Estimated time: ~15-20 minutes")
    print("\nPress Enter to continue, or Ctrl+C to cancel...")
    input()
    
    # Collect traffic data
    collect_traffic_data(waypoints, days=7)
    
    print("\n✓ Traffic data saved to traffic_data/")


if __name__ == "__main__":
    main()
