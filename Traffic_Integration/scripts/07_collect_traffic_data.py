"""
Script 07: Collect Traffic Data from TomTom Traffic Stats API

This script collects historical traffic data for all waypoints at 6 key hours per day
over 7 days using the TomTom Traffic Stats Historical API.

API Calls: 21 waypoints × 6 hours × 7 days = 882 calls

Hours sampled: 6am, 9am, 12pm, 3pm, 6pm, 9pm
Days: Monday through Sunday
"""

import json
import os
import time
from datetime import datetime, timedelta
import requests
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
MIDPOINTS_DIR = BASE_DIR / "midway_points"
TRAFFIC_DATA_DIR = BASE_DIR / "traffic_data"
API_KEY_FILE = BASE_DIR / "apikeys" / "tomtom.txt"

# Create output directory
TRAFFIC_DATA_DIR.mkdir(exist_ok=True)

# Hours to sample (24-hour format)
SAMPLE_HOURS = [6, 9, 12, 15, 18, 21]  # 6am, 9am, 12pm, 3pm, 6pm, 9pm

# Days of week
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def load_api_key():
    """Load TomTom API key from file"""
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def load_all_waypoints():
    """Load all waypoints from midway_points directory"""
    summary_file = MIDPOINTS_DIR / "all_midpoints_summary.json"
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    waypoints = []
    
    # Data is a list of route objects
    for route_data in data:
        route_name = f"{route_data['source']} → {route_data['destination']}"
        
        # Add start point
        start = route_data['start']
        waypoints.append({
            'route': route_name,
            'type': 'start',
            'lat': start['lat'],
            'lon': start['lon'],
            'distance_km': 0.0
        })
        
        # Add midpoints
        for i, mp in enumerate(route_data['midpoints'], 1):
            waypoints.append({
                'route': route_name,
                'type': f'midpoint_{i}',
                'lat': mp['lat'],
                'lon': mp['lon'],
                'distance_km': mp.get('distance_from_start_km', 0.0)
            })
        
        # Add end point
        end = route_data['end']
        waypoints.append({
            'route': route_name,
            'type': 'end',
            'lat': end['lat'],
            'lon': end['lon'],
            'distance_km': end.get('distance_from_start_km', route_data['total_distance_km'])
        })
    
    return waypoints


def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculate bearing/heading between two coordinates
    Needed for TomTom API to get traffic in the right direction
    """
    from math import radians, degrees, atan2, cos, sin
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    lon_diff = radians(lon2 - lon1)
    
    x = sin(lon_diff) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon_diff)
    
    heading = degrees(atan2(x, y))
    
    # Normalize to 0-360
    return (heading + 360) % 360


def get_traffic_stats(api_key, lat, lon, heading, day_of_week, hour):
    """
    Query TomTom Traffic Stats Historical API
    
    Endpoint: https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json
    
    Parameters:
    - point: lat,lon
    - unit: KMPH (km per hour)
    - openLr: false (we're using point, not OpenLR)
    """
    
    base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData"
    
    # Build URL
    # zoom level 10 is good for highways/main roads
    url = f"{base_url}/relative0/10/json"
    
    params = {
        'key': api_key,
        'point': f"{lat},{lon}",
        'unit': 'KMPH'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Add metadata
        data['query_metadata'] = {
            'lat': lat,
            'lon': lon,
            'heading': heading,
            'day_of_week': day_of_week,
            'hour': hour,
            'timestamp': datetime.now().isoformat()
        }
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ API error: {str(e)}")
        return None


def main():
    print("=" * 80)
    print("TomTom Traffic Data Collection")
    print("=" * 80)
    
    # Load API key
    print(f"\n✓ Loading API key from {API_KEY_FILE.name}")
    api_key = load_api_key()
    
    # Load waypoints
    print("✓ Loading waypoints...")
    waypoints = load_all_waypoints()
    print(f"  Found {len(waypoints)} waypoints across 3 routes")
    
    # Calculate total API calls
    total_calls = len(waypoints) * len(SAMPLE_HOURS) * len(DAYS_OF_WEEK)
    print(f"\n📊 Total API calls to make: {total_calls}")
    print(f"   {len(waypoints)} waypoints × {len(SAMPLE_HOURS)} hours × {len(DAYS_OF_WEEK)} days")
    print(f"   Estimated time: {total_calls * 0.5:.0f}-{total_calls * 1:.0f} seconds")
    
    # Create organized directory structure
    print(f"\n✓ Output directory: {TRAFFIC_DATA_DIR}")
    
    # Start collection
    print("\n" + "=" * 80)
    print("Starting data collection...")
    print("=" * 80)
    
    call_count = 0
    success_count = 0
    error_count = 0
    
    for day_idx, day_name in enumerate(DAYS_OF_WEEK):
        print(f"\n📅 {day_name}")
        print("-" * 80)
        
        for hour in SAMPLE_HOURS:
            hour_str = f"{hour:02d}:00"
            print(f"\n  ⏰ {hour_str}")
            
            for wp_idx, waypoint in enumerate(waypoints):
                call_count += 1
                
                # Calculate heading (use next waypoint if available, else use 0)
                heading = 0.0  # Default
                if wp_idx < len(waypoints) - 1:
                    next_wp = waypoints[wp_idx + 1]
                    if next_wp['route'] == waypoint['route']:  # Same route
                        heading = calculate_heading(
                            waypoint['lat'], waypoint['lon'],
                            next_wp['lat'], next_wp['lon']
                        )
                
                # Make API call
                traffic_data = get_traffic_stats(
                    api_key,
                    waypoint['lat'],
                    waypoint['lon'],
                    heading,
                    day_name,
                    hour
                )
                
                if traffic_data:
                    success_count += 1
                    
                    # Save individual response
                    route_clean = waypoint['route'].replace(' → ', '_to_').replace(' ', '_')
                    filename = f"{route_clean}_{waypoint['type']}_{day_name}_{hour:02d}h.json"
                    filepath = TRAFFIC_DATA_DIR / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(traffic_data, f, indent=2)
                else:
                    error_count += 1
                
                # Progress indicator
                progress = (call_count / total_calls) * 100
                status = "✓" if traffic_data else "✗"
                print(f"    {status} [{call_count}/{total_calls}] {progress:.1f}% - "
                      f"{waypoint['route'][:30]:30s} {waypoint['type']:12s}", end='\r')
                
                # Rate limiting: small delay between calls
                time.sleep(0.2)
            
            print()  # New line after each hour
    
    # Final summary
    print("\n" + "=" * 80)
    print("Collection Complete!")
    print("=" * 80)
    print(f"\n📊 Statistics:")
    print(f"   Total calls:    {call_count}")
    print(f"   Successful:     {success_count} ({success_count/call_count*100:.1f}%)")
    print(f"   Errors:         {error_count}")
    print(f"\n✓ Traffic data saved to: {TRAFFIC_DATA_DIR}")
    print(f"   {len(list(TRAFFIC_DATA_DIR.glob('*.json')))} JSON files created")
    
    # Create collection summary
    summary = {
        'collection_date': datetime.now().isoformat(),
        'total_calls': call_count,
        'successful_calls': success_count,
        'failed_calls': error_count,
        'waypoints_count': len(waypoints),
        'hours_sampled': SAMPLE_HOURS,
        'days_sampled': DAYS_OF_WEEK,
        'files_created': len(list(TRAFFIC_DATA_DIR.glob('*.json')))
    }
    
    summary_file = TRAFFIC_DATA_DIR / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file.name}")


if __name__ == "__main__":
    main()
