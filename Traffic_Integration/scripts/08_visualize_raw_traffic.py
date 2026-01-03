"""
Script 08: Visualize Raw Traffic Data

Creates interactive maps showing traffic conditions at each waypoint
across different times and days.

Maps created:
1. traffic_overview_map.html - All waypoints with traffic data
2. traffic_timeline_map.html - Hour-by-hour comparison
3. traffic_heatmap.html - Congestion heatmap
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
MIDPOINTS_DIR = BASE_DIR / "midway_points"
MAPS_DIR = BASE_DIR / "maps"
SELECTED_ROUTES_DIR = BASE_DIR / "selected_routes"

# Create maps directory
MAPS_DIR.mkdir(exist_ok=True)


def load_traffic_data():
    """Load all traffic data files and organize by waypoint, day, hour"""
    traffic_files = list(TRAFFIC_DATA_DIR.glob("*.json"))
    
    if not traffic_files:
        print("⚠ No traffic data files found!")
        return {}
    
    print(f"✓ Found {len(traffic_files)} traffic data files")
    
    # Organize data: {waypoint_id: {day: {hour: data}}}
    organized_data = {}
    
    for filepath in traffic_files:
        # Skip summary file
        if 'summary' in filepath.name.lower():
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('query_metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        day = metadata.get('day_of_week')
        hour = metadata.get('hour')
        
        # Skip if missing required metadata
        if lat is None or lon is None or day is None or hour is None:
            continue
        
        # Create unique waypoint ID
        wp_id = f"{lat:.5f},{lon:.5f}"
        
        if wp_id not in organized_data:
            organized_data[wp_id] = {
                'lat': lat,
                'lon': lon,
                'data': {}
            }
        
        if day not in organized_data[wp_id]['data']:
            organized_data[wp_id]['data'][day] = {}
        
        organized_data[wp_id]['data'][day][hour] = data
    
    return organized_data


def extract_speed_info(traffic_response):
    """Extract current and free flow speeds from TomTom response"""
    flow_data = traffic_response.get('flowSegmentData', {})
    
    current_speed = flow_data.get('currentSpeed', 0)
    free_flow_speed = flow_data.get('freeFlowSpeed', 0)
    confidence = flow_data.get('confidence', 0)
    
    # Calculate congestion ratio (0 = free flow, 1 = stopped)
    if free_flow_speed > 0:
        congestion = 1 - (current_speed / free_flow_speed)
    else:
        congestion = 0
    
    return {
        'current_speed': current_speed,
        'free_flow_speed': free_flow_speed,
        'confidence': confidence,
        'congestion': max(0, min(1, congestion))  # Clamp 0-1
    }


def get_congestion_color(congestion_level):
    """
    Return color based on congestion level
    0.0 = green (free flow)
    0.3 = yellow (moderate)
    0.6 = orange (heavy)
    1.0 = red (stopped)
    """
    if congestion_level < 0.2:
        return '#2ecc71'  # Green
    elif congestion_level < 0.4:
        return '#f1c40f'  # Yellow
    elif congestion_level < 0.6:
        return '#e67e22'  # Orange
    else:
        return '#e74c3c'  # Red


def create_traffic_overview_map(traffic_data):
    """
    Create overview map showing average traffic conditions at each waypoint
    """
    print("\n📍 Creating traffic overview map...")
    
    # Calculate center point
    all_lats = [wp['lat'] for wp in traffic_data.values()]
    all_lons = [wp['lon'] for wp in traffic_data.values()]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Process each waypoint
    for wp_id, wp_info in traffic_data.items():
        lat = wp_info['lat']
        lon = wp_info['lon']
        
        # Calculate average congestion across all days/hours
        all_congestion = []
        all_current_speeds = []
        all_free_flow_speeds = []
        
        for day_data in wp_info['data'].values():
            for hour_data in day_data.values():
                speed_info = extract_speed_info(hour_data)
                all_congestion.append(speed_info['congestion'])
                all_current_speeds.append(speed_info['current_speed'])
                all_free_flow_speeds.append(speed_info['free_flow_speed'])
        
        if all_congestion:
            avg_congestion = statistics.mean(all_congestion)
            avg_current_speed = statistics.mean(all_current_speeds)
            avg_free_flow_speed = statistics.mean(all_free_flow_speeds)
            
            # Create marker
            color = get_congestion_color(avg_congestion)
            
            # Popup with details
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>Waypoint Traffic</h4>
                <b>Location:</b> {lat:.5f}, {lon:.5f}<br>
                <b>Avg Current Speed:</b> {avg_current_speed:.1f} km/h<br>
                <b>Avg Free Flow Speed:</b> {avg_free_flow_speed:.1f} km/h<br>
                <b>Avg Congestion:</b> {avg_congestion*100:.1f}%<br>
                <b>Data Points:</b> {len(all_congestion)}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px">
    <p><b>Traffic Congestion</b></p>
    <p><i class="fa fa-circle" style="color:#2ecc71"></i> Free Flow (0-20%)</p>
    <p><i class="fa fa-circle" style="color:#f1c40f"></i> Moderate (20-40%)</p>
    <p><i class="fa fa-circle" style="color:#e67e22"></i> Heavy (40-60%)</p>
    <p><i class="fa fa-circle" style="color:#e74c3c"></i> Congested (60%+)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_file = MAPS_DIR / "traffic_overview_map.html"
    m.save(str(output_file))
    print(f"✓ Saved: {output_file.name}")
    
    return output_file


def create_hourly_comparison_maps(traffic_data):
    """
    Create maps for each sample hour showing traffic conditions
    """
    print("\n⏰ Creating hourly comparison maps...")
    
    hours = [6, 9, 12, 15, 18, 21]
    
    # Calculate center
    all_lats = [wp['lat'] for wp in traffic_data.values()]
    all_lons = [wp['lon'] for wp in traffic_data.values()]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    for hour in hours:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = f"""
        <div style="position: fixed; top: 10px; left: 50px; width: 300px; height: 50px; 
                    background-color: white; border:2px solid grey; z-index:9999; font-size:16px;
                    padding: 10px; text-align: center;">
        <b>Traffic Conditions at {hour:02d}:00</b><br>
        <small>Average across all days</small>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Process each waypoint for this hour
        for wp_id, wp_info in traffic_data.items():
            lat = wp_info['lat']
            lon = wp_info['lon']
            
            # Collect data for this hour across all days
            hour_congestion = []
            hour_speeds = []
            
            for day_data in wp_info['data'].values():
                if hour in day_data:
                    speed_info = extract_speed_info(day_data[hour])
                    hour_congestion.append(speed_info['congestion'])
                    hour_speeds.append(speed_info['current_speed'])
            
            if hour_congestion:
                avg_congestion = statistics.mean(hour_congestion)
                avg_speed = statistics.mean(hour_speeds)
                
                color = get_congestion_color(avg_congestion)
                
                popup_html = f"""
                <div style="font-family: Arial; width: 180px;">
                    <b>Hour:</b> {hour:02d}:00<br>
                    <b>Avg Speed:</b> {avg_speed:.1f} km/h<br>
                    <b>Congestion:</b> {avg_congestion*100:.1f}%
                </div>
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 150px; 
                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                    padding: 10px">
        <p><b>Congestion Level</b></p>
        <p><i class="fa fa-circle" style="color:#2ecc71"></i> Free Flow</p>
        <p><i class="fa fa-circle" style="color:#f1c40f"></i> Moderate</p>
        <p><i class="fa fa-circle" style="color:#e67e22"></i> Heavy</p>
        <p><i class="fa fa-circle" style="color:#e74c3c"></i> Congested</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save
        output_file = MAPS_DIR / f"traffic_hour_{hour:02d}h.html"
        m.save(str(output_file))
        print(f"  ✓ {hour:02d}:00 → {output_file.name}")


def main():
    print("=" * 80)
    print("Traffic Data Visualization")
    print("=" * 80)
    
    # Load traffic data
    print("\n📊 Loading traffic data...")
    traffic_data = load_traffic_data()
    
    if not traffic_data:
        print("❌ No traffic data to visualize. Run script 07 first.")
        return
    
    print(f"✓ Loaded data for {len(traffic_data)} waypoints")
    
    # Create overview map
    create_traffic_overview_map(traffic_data)
    
    # Create hourly comparison maps
    create_hourly_comparison_maps(traffic_data)
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\n✓ Maps saved to: {MAPS_DIR}")
    print("\nGenerated maps:")
    print("  1. traffic_overview_map.html - Overall traffic patterns")
    print("  2. traffic_hour_06h.html through traffic_hour_21h.html - Hourly views")


if __name__ == "__main__":
    main()
