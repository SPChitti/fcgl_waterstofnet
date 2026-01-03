"""
Script 09: Peak vs Off-Peak Traffic Comparison

Creates side-by-side comparison maps showing:
1. Morning peak (9am) vs Night (9pm)
2. Evening peak (6pm) vs Early morning (6am)
3. Weekday average vs Weekend average
4. Congestion heatmap

Also generates statistics on travel time differences.
"""

import json
import os
from pathlib import Path
import folium
from folium import plugins
import statistics

# Configuration
BASE_DIR = Path(__file__).parent.parent
TRAFFIC_DATA_DIR = BASE_DIR / "traffic_data"
SELECTED_ROUTES_DIR = BASE_DIR / "selected_routes"
MAPS_DIR = BASE_DIR / "maps"

MAPS_DIR.mkdir(exist_ok=True)


def load_traffic_data():
    """Load and organize traffic data"""
    traffic_files = list(TRAFFIC_DATA_DIR.glob("*.json"))
    
    if not traffic_files:
        return {}
    
    organized_data = {}
    
    for filepath in traffic_files:
        # Skip summary files
        if 'summary' in filepath.name.lower():
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Parse filename to extract route, waypoint type, day, hour
        # Format: RouteAtoB_waypointtype_Day_HHh.json
        filename = filepath.stem
        parts = filename.split('_')
        
        # Extract metadata
        metadata = data.get('query_metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        day = metadata.get('day_of_week')
        hour = metadata.get('hour')
        
        # Skip if missing required metadata
        if lat is None or lon is None or day is None or hour is None:
            continue
        
        # Determine route and waypoint type from filename
        route_parts = []
        wp_type = None
        for i, part in enumerate(parts):
            if 'to' in part:
                route_parts.append(part)
            elif part in ['start', 'end'] or 'midpoint' in part:
                wp_type = part if part in ['start', 'end'] else '_'.join(parts[i:i+2])
                break
            else:
                route_parts.append(part)
        
        route_name = '_'.join(route_parts)
        
        # Create unique ID
        wp_id = f"{route_name}_{wp_type}"
        
        if wp_id not in organized_data:
            organized_data[wp_id] = {
                'route': route_name,
                'type': wp_type,
                'lat': lat,
                'lon': lon,
                'data': {}
            }
        
        if day not in organized_data[wp_id]['data']:
            organized_data[wp_id]['data'][day] = {}
        
        organized_data[wp_id]['data'][day][hour] = data
    
    return organized_data


def extract_speed_info(traffic_response):
    """Extract speed information from TomTom response"""
    flow_data = traffic_response.get('flowSegmentData', {})
    
    current_speed = flow_data.get('currentSpeed', 0)
    free_flow_speed = flow_data.get('freeFlowSpeed', 0)
    
    if free_flow_speed > 0:
        congestion = 1 - (current_speed / free_flow_speed)
    else:
        congestion = 0
    
    return {
        'current_speed': current_speed,
        'free_flow_speed': free_flow_speed,
        'congestion': max(0, min(1, congestion)),
        'speed_ratio': current_speed / free_flow_speed if free_flow_speed > 0 else 1.0
    }


def get_congestion_color(congestion_level):
    """Return color based on congestion level"""
    if congestion_level < 0.2:
        return '#2ecc71'  # Green
    elif congestion_level < 0.4:
        return '#f1c40f'  # Yellow
    elif congestion_level < 0.6:
        return '#e67e22'  # Orange
    else:
        return '#e74c3c'  # Red


def create_peak_offpeak_comparison(traffic_data, peak_hour, offpeak_hour, title):
    """
    Create side-by-side comparison map for peak vs off-peak
    """
    print(f"\n📊 Creating {title}...")
    
    # Calculate center
    all_lats = [wp['lat'] for wp in traffic_data.values() if wp['lat'] is not None]
    all_lons = [wp['lon'] for wp in traffic_data.values() if wp['lon'] is not None]
    
    if not all_lats or not all_lons:
        print("⚠ No valid coordinates found!")
        return None
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Create feature groups for peak and off-peak
    peak_group = folium.FeatureGroup(name=f'Peak Hour ({peak_hour:02d}:00)', show=True)
    offpeak_group = folium.FeatureGroup(name=f'Off-Peak Hour ({offpeak_hour:02d}:00)', show=True)
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Process each waypoint
    for wp_id, wp_info in traffic_data.items():
        lat = wp_info['lat']
        lon = wp_info['lon']
        
        # Collect peak hour data (weekdays average)
        peak_speeds = []
        peak_congestion = []
        
        for day in weekdays:
            if day in wp_info['data'] and peak_hour in wp_info['data'][day]:
                speed_info = extract_speed_info(wp_info['data'][day][peak_hour])
                peak_speeds.append(speed_info['current_speed'])
                peak_congestion.append(speed_info['congestion'])
        
        # Collect off-peak hour data
        offpeak_speeds = []
        offpeak_congestion = []
        
        for day in weekdays:
            if day in wp_info['data'] and offpeak_hour in wp_info['data'][day]:
                speed_info = extract_speed_info(wp_info['data'][day][offpeak_hour])
                offpeak_speeds.append(speed_info['current_speed'])
                offpeak_congestion.append(speed_info['congestion'])
        
        if peak_speeds and offpeak_speeds:
            avg_peak_speed = statistics.mean(peak_speeds)
            avg_peak_congestion = statistics.mean(peak_congestion)
            
            avg_offpeak_speed = statistics.mean(offpeak_speeds)
            avg_offpeak_congestion = statistics.mean(offpeak_congestion)
            
            speed_diff = avg_offpeak_speed - avg_peak_speed
            time_increase_pct = ((1/avg_peak_speed) - (1/avg_offpeak_speed)) / (1/avg_offpeak_speed) * 100 if avg_peak_speed > 0 else 0
            
            # Peak hour marker (slightly offset to left)
            peak_color = get_congestion_color(avg_peak_congestion)
            peak_popup = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>{peak_hour:02d}:00 (Peak)</h4>
                <b>Avg Speed:</b> {avg_peak_speed:.1f} km/h<br>
                <b>Congestion:</b> {avg_peak_congestion*100:.1f}%<br>
                <b>Speed Difference:</b> {speed_diff:+.1f} km/h<br>
                <b>Time Increase:</b> {time_increase_pct:+.1f}%
            </div>
            """
            
            folium.CircleMarker(
                location=[lat - 0.005, lon - 0.005],  # Slight offset
                radius=10,
                color=peak_color,
                fill=True,
                fillColor=peak_color,
                fillOpacity=0.7,
                popup=folium.Popup(peak_popup, max_width=300)
            ).add_to(peak_group)
            
            # Off-peak hour marker (slightly offset to right)
            offpeak_color = get_congestion_color(avg_offpeak_congestion)
            offpeak_popup = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>{offpeak_hour:02d}:00 (Off-Peak)</h4>
                <b>Avg Speed:</b> {avg_offpeak_speed:.1f} km/h<br>
                <b>Congestion:</b> {avg_offpeak_congestion*100:.1f}%
            </div>
            """
            
            folium.CircleMarker(
                location=[lat + 0.005, lon + 0.005],  # Slight offset
                radius=10,
                color=offpeak_color,
                fill=True,
                fillColor=offpeak_color,
                fillOpacity=0.7,
                popup=folium.Popup(offpeak_popup, max_width=300)
            ).add_to(offpeak_group)
    
    peak_group.add_to(m)
    offpeak_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; width: 400px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:16px;
                padding: 10px; text-align: center;">
    <b>{title}</b><br>
    <small>Peak hour markers on left, Off-peak on right</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <p><b>Congestion Level</b></p>
    <p><i class="fa fa-circle" style="color:#2ecc71"></i> Free Flow (0-20%)</p>
    <p><i class="fa fa-circle" style="color:#f1c40f"></i> Moderate (20-40%)</p>
    <p><i class="fa fa-circle" style="color:#e67e22"></i> Heavy (40-60%)</p>
    <p><i class="fa fa-circle" style="color:#e74c3c"></i> Congested (60%+)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('vs', 'vs') + '.html'
    output_file = MAPS_DIR / filename
    m.save(str(output_file))
    print(f"✓ Saved: {output_file.name}")
    
    return output_file


def calculate_route_time_comparison(traffic_data):
    """
    Calculate estimated travel time for each route at different hours
    """
    print("\n⏱️  Calculating travel time comparisons...")
    
    hours = [6, 9, 12, 15, 18, 21]
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Load route distances
    routes_info = {}
    for route_file in SELECTED_ROUTES_DIR.glob("*.json"):
        if 'summary' in route_file.name:
            continue
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        # Build route name from source and destination
        source = route_data.get('source', '')
        destination = route_data.get('destination', '')
        route_name = f"{source}_to_{destination}"
        
        # Get distance from path
        distance_m = route_data['routes'][0]['path']['distance']
        distance_km = distance_m / 1000
        
        routes_info[route_name] = {
            'distance_km': distance_km,
            'travel_times': {}
        }
    
    # Calculate travel times for each hour
    for hour in hours:
        for route_key, route_info in routes_info.items():
            # Find all waypoints for this route
            route_waypoints = [wp for wp_id, wp in traffic_data.items() if route_key in wp_id]
            
            if not route_waypoints:
                continue
            
            # Calculate average speed at this hour
            hour_speeds = []
            
            for wp in route_waypoints:
                for day in weekdays:
                    if day in wp['data'] and hour in wp['data'][day]:
                        speed_info = extract_speed_info(wp['data'][day][hour])
                        if speed_info['current_speed'] > 0:
                            hour_speeds.append(speed_info['current_speed'])
            
            if hour_speeds:
                avg_speed = statistics.mean(hour_speeds)
                travel_time_hours = route_info['distance_km'] / avg_speed
                travel_time_minutes = travel_time_hours * 60
                
                route_info['travel_times'][hour] = {
                    'avg_speed_kmh': avg_speed,
                    'travel_time_minutes': travel_time_minutes
                }
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Travel Time Comparison Report")
    report_lines.append("=" * 80)
    
    for route_name, route_info in routes_info.items():
        if not route_info['travel_times']:
            continue
        
        report_lines.append(f"\n{route_name.replace('_', ' → ')}")
        report_lines.append(f"Distance: {route_info['distance_km']:.1f} km")
        report_lines.append("-" * 80)
        
        # Find best and worst times
        times = route_info['travel_times']
        best_hour = min(times.keys(), key=lambda h: times[h]['travel_time_minutes'])
        worst_hour = max(times.keys(), key=lambda h: times[h]['travel_time_minutes'])
        
        best_time = times[best_hour]['travel_time_minutes']
        worst_time = times[worst_hour]['travel_time_minutes']
        time_diff = worst_time - best_time
        time_diff_pct = (time_diff / best_time) * 100
        
        report_lines.append(f"Best time:  {best_hour:02d}:00 - {best_time:.1f} min ({times[best_hour]['avg_speed_kmh']:.1f} km/h)")
        report_lines.append(f"Worst time: {worst_hour:02d}:00 - {worst_time:.1f} min ({times[worst_hour]['avg_speed_kmh']:.1f} km/h)")
        report_lines.append(f"Difference: +{time_diff:.1f} min ({time_diff_pct:+.1f}%)")
        
        report_lines.append("\nHourly breakdown:")
        for hour in sorted(times.keys()):
            t = times[hour]
            report_lines.append(f"  {hour:02d}:00 - {t['travel_time_minutes']:5.1f} min @ {t['avg_speed_kmh']:5.1f} km/h")
    
    report_lines.append("\n" + "=" * 80)
    
    # Save report
    report_file = MAPS_DIR.parent / "traffic_data" / "travel_time_comparison.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report saved: {report_file.name}")
    
    # Print summary
    print("\n" + '\n'.join(report_lines))


def main():
    print("=" * 80)
    print("Peak vs Off-Peak Traffic Comparison")
    print("=" * 80)
    
    # Load traffic data
    print("\n📊 Loading traffic data...")
    traffic_data = load_traffic_data()
    
    if not traffic_data:
        print("❌ No traffic data found. Run script 07 first.")
        return
    
    print(f"✓ Loaded data for {len(traffic_data)} waypoints")
    
    # Create comparison maps
    create_peak_offpeak_comparison(
        traffic_data,
        peak_hour=9,
        offpeak_hour=21,
        title="Morning Peak (09:00) vs Night (21:00)"
    )
    
    create_peak_offpeak_comparison(
        traffic_data,
        peak_hour=18,
        offpeak_hour=6,
        title="Evening Peak (18:00) vs Early Morning (06:00)"
    )
    
    # Calculate travel time comparisons
    calculate_route_time_comparison(traffic_data)
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print(f"\n✓ Maps saved to: {MAPS_DIR}")
    print("\nGenerated comparison maps:")
    print("  1. morning_peak_09_00_vs_night_21_00.html")
    print("  2. evening_peak_18_00_vs_early_morning_06_00.html")
    print("\n✓ Travel time report: traffic_data/travel_time_comparison.txt")


if __name__ == "__main__":
    main()
