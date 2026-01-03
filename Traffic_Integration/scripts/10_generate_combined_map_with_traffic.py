"""
Script 10: Generate Combined Features Map with Traffic Data

Creates an enhanced combined map with all route features AND traffic data layers.

Layers included:
1. Base Routes
2. Speed Limits
3. Road Classes
4. Slopes
5. Traffic Overview (average)
6. Traffic - Morning Peak (9am)
7. Traffic - Evening Peak (6pm)
8. Traffic - Off-Peak (9pm)
"""

import json
import os
from pathlib import Path
import folium
from folium import plugins
import statistics

# Configuration
BASE_DIR = Path(__file__).parent.parent
SELECTED_ROUTES_DIR = BASE_DIR / "selected_routes"
ROUTE_FEATURES_DIR = BASE_DIR / "route_features"
MIDPOINTS_DIR = BASE_DIR / "midway_points"
TRAFFIC_DATA_DIR = BASE_DIR / "traffic_data"
MAPS_DIR = BASE_DIR / "maps"

MAPS_DIR.mkdir(exist_ok=True)


def load_traffic_data():
    """Load and organize traffic data"""
    traffic_files = list(TRAFFIC_DATA_DIR.glob("*.json"))
    
    if not traffic_files:
        return {}
    
    organized_data = {}
    
    for filepath in traffic_files:
        if 'summary' in filepath.name.lower():
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('query_metadata', {})
        lat = metadata.get('lat')
        lon = metadata.get('lon')
        day = metadata.get('day_of_week')
        hour = metadata.get('hour')
        
        if lat is None or lon is None or day is None or hour is None:
            continue
        
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
        'congestion': max(0, min(1, congestion))
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


def add_traffic_layer(m, traffic_data, layer_name, hour=None, description=""):
    """
    Add a traffic layer to the map
    
    Args:
        m: Folium map object
        traffic_data: Organized traffic data
        layer_name: Name of the layer
        hour: Specific hour to show (None = average across all hours)
        description: Description for the layer
    """
    layer = folium.FeatureGroup(name=layer_name, show=False)
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    for wp_id, wp_info in traffic_data.items():
        lat = wp_info['lat']
        lon = wp_info['lon']
        
        speeds = []
        congestion_levels = []
        
        # Collect data for specified hour or all hours
        for day in weekdays:
            if day not in wp_info['data']:
                continue
            
            if hour is not None:
                # Specific hour
                if hour in wp_info['data'][day]:
                    speed_info = extract_speed_info(wp_info['data'][day][hour])
                    speeds.append(speed_info['current_speed'])
                    congestion_levels.append(speed_info['congestion'])
            else:
                # All hours average
                for h in [6, 9, 12, 15, 18, 21]:
                    if h in wp_info['data'][day]:
                        speed_info = extract_speed_info(wp_info['data'][day][h])
                        speeds.append(speed_info['current_speed'])
                        congestion_levels.append(speed_info['congestion'])
        
        if speeds:
            avg_speed = statistics.mean(speeds)
            avg_congestion = statistics.mean(congestion_levels)
            
            color = get_congestion_color(avg_congestion)
            
            hour_str = f"{hour:02d}:00" if hour is not None else "All Hours"
            
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>{layer_name}</h4>
                <b>Time:</b> {hour_str}<br>
                <b>Avg Speed:</b> {avg_speed:.1f} km/h<br>
                <b>Congestion:</b> {avg_congestion*100:.1f}%<br>
                <b>Data Points:</b> {len(speeds)}
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                weight=2
            ).add_to(layer)
    
    layer.add_to(m)
    return layer


def create_combined_map_with_traffic():
    """Create combined map with all features and traffic data"""
    print("\n📍 Creating combined map with traffic data...")
    
    # Load route data
    route_files = [f for f in SELECTED_ROUTES_DIR.glob("*.json") if 'summary' not in f.name]
    
    if not route_files:
        print("❌ No route files found!")
        return
    
    # Store routes with their filenames for matching
    all_routes = {}
    for route_file in sorted(route_files):
        with open(route_file, 'r') as f:
            route_key = route_file.stem  # e.g., "Dendermonde_to_Mechelen_20251229_174249"
            all_routes[route_key] = json.load(f)
    
    # Calculate center
    all_lats = []
    all_lons = []
    
    for route_key, route in all_routes.items():
        coords = route['routes'][0]['path']['points']['coordinates']
        for coord in coords:
            all_lons.append(coord[0])
            all_lats.append(coord[1])
    
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50px; width: 500px; height: 70px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:16px;
                padding: 10px; text-align: center;">
    <b>Combined Route Features + Traffic Data</b><br>
    <small>Use layer control (top right) to toggle features</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # ========== LAYER 1: Base Routes ==========
    base_group = folium.FeatureGroup(name='Base Routes', show=True)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (route_key, route) in enumerate(all_routes.items()):
        source = route['source']
        dest = route['destination']
        coords = route['routes'][0]['path']['points']['coordinates']
        
        route_coords = [[coord[1], coord[0]] for coord in coords]
        
        folium.PolyLine(
            route_coords,
            color=colors[idx % len(colors)],
            weight=4,
            opacity=0.7,
            popup=f"{source} → {dest}"
        ).add_to(base_group)
    
    base_group.add_to(m)
    
    # ========== LAYER 2: Speed Limits ==========
    speed_group = folium.FeatureGroup(name='Speed Limits', show=False)
    
    for route_file in sorted(ROUTE_FEATURES_DIR.glob("*_features.json")):
        with open(route_file, 'r') as f:
            features = json.load(f)
        
        route_key = route_file.stem.replace('_features', '')
        route_data = all_routes.get(route_key)
        
        if not route_data:
            continue
        
        coords = route_data['routes'][0]['path']['points']['coordinates']
        max_speeds = features['max_speed']
        
        # Create index to speed mapping
        speed_map = {}
        for speed_detail in max_speeds:
            for i in range(speed_detail['start_index'], speed_detail['end_index'] + 1):
                if i < len(coords):
                    speed_map[i] = speed_detail.get('value')
        
        # Group consecutive indices with same speed
        if speed_map:
            current_speed = None
            segment_indices = []
            
            for i in sorted(speed_map.keys()):
                speed = speed_map[i]
                
                if speed != current_speed:
                    if segment_indices:
                        segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                        
                        if current_speed is None or current_speed == 0:
                            color = '#95a5a6'
                        elif current_speed <= 50:
                            color = '#e74c3c'
                        elif current_speed <= 70:
                            color = '#f39c12'
                        elif current_speed <= 90:
                            color = '#f1c40f'
                        elif current_speed <= 110:
                            color = '#3498db'
                        else:
                            color = '#2ecc71'
                        
                        folium.PolyLine(
                            segment_coords,
                            color=color,
                            weight=4,
                            opacity=0.7,
                            popup=f"Speed: {current_speed} km/h" if current_speed else "Unknown"
                        ).add_to(speed_group)
                    
                    segment_indices = [i]
                    current_speed = speed
                else:
                    segment_indices.append(i)
            
            # Last segment
            if segment_indices:
                segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                
                if current_speed is None or current_speed == 0:
                    color = '#95a5a6'
                elif current_speed <= 50:
                    color = '#e74c3c'
                elif current_speed <= 70:
                    color = '#f39c12'
                elif current_speed <= 90:
                    color = '#f1c40f'
                elif current_speed <= 110:
                    color = '#3498db'
                else:
                    color = '#2ecc71'
                
                folium.PolyLine(
                    segment_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Speed: {current_speed} km/h" if current_speed else "Unknown"
                ).add_to(speed_group)
    
    speed_group.add_to(m)
    
    # ========== LAYER 3: Road Classes ==========
    road_group = folium.FeatureGroup(name='Road Classes', show=False)
    
    road_class_colors = {
        'motorway': '#e74c3c',
        'trunk': '#e67e22',
        'primary': '#f39c12',
        'secondary': '#f1c40f',
        'tertiary': '#3498db',
        'residential': '#95a5a6',
        'service': '#bdc3c7'
    }
    
    for route_file in sorted(ROUTE_FEATURES_DIR.glob("*_features.json")):
        with open(route_file, 'r') as f:
            features = json.load(f)
        
        route_key = route_file.stem.replace('_features', '')
        route_data = all_routes.get(route_key)
        
        if not route_data:
            continue
        
        coords = route_data['routes'][0]['path']['points']['coordinates']
        road_classes = features['road_class']
        
        # Create index to class mapping
        class_map = {}
        for class_detail in road_classes:
            for i in range(class_detail['start_index'], class_detail['end_index'] + 1):
                if i < len(coords):
                    class_map[i] = class_detail.get('value')
        
        if class_map:
            current_class = None
            segment_indices = []
            
            for i in sorted(class_map.keys()):
                road_class = class_map[i]
                
                if road_class != current_class:
                    if segment_indices:
                        segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                        color = road_class_colors.get(current_class, '#95a5a6')
                        
                        folium.PolyLine(
                            segment_coords,
                            color=color,
                            weight=4,
                            opacity=0.7,
                            popup=f"Class: {current_class}"
                        ).add_to(road_group)
                    
                    segment_indices = [i]
                    current_class = road_class
                else:
                    segment_indices.append(i)
            
            if segment_indices:
                segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                color = road_class_colors.get(current_class, '#95a5a6')
                
                folium.PolyLine(
                    segment_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Class: {current_class}"
                ).add_to(road_group)
    
    road_group.add_to(m)
    
    # ========== LAYER 4: Slopes ==========
    slope_group = folium.FeatureGroup(name='Slopes/Gradients', show=False)
    
    for route_file in sorted(ROUTE_FEATURES_DIR.glob("*_features.json")):
        with open(route_file, 'r') as f:
            features = json.load(f)
        
        route_key = route_file.stem.replace('_features', '')
        route_data = all_routes.get(route_key)
        
        if not route_data:
            continue
        
        coords = route_data['routes'][0]['path']['points']['coordinates']
        slopes = features['average_slope']
        
        slope_map = {}
        for slope_detail in slopes:
            for i in range(slope_detail['start_index'], slope_detail['end_index'] + 1):
                if i < len(coords):
                    slope_map[i] = slope_detail.get('value')
        
        if slope_map:
            current_slope = None
            segment_indices = []
            
            for i in sorted(slope_map.keys()):
                slope = slope_map[i]
                
                if slope != current_slope:
                    if segment_indices:
                        segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                        
                        if current_slope < -5:
                            color = '#3498db'
                        elif current_slope < -2:
                            color = '#5dade2'
                        elif current_slope < 2:
                            color = '#2ecc71'
                        elif current_slope < 5:
                            color = '#f39c12'
                        else:
                            color = '#e74c3c'
                        
                        folium.PolyLine(
                            segment_coords,
                            color=color,
                            weight=4,
                            opacity=0.7,
                            popup=f"Slope: {current_slope:.1f}%"
                        ).add_to(slope_group)
                    
                    segment_indices = [i]
                    current_slope = slope
                else:
                    segment_indices.append(i)
            
            if segment_indices:
                segment_coords = [[coords[idx][1], coords[idx][0]] for idx in segment_indices]
                
                if current_slope < -5:
                    color = '#3498db'
                elif current_slope < -2:
                    color = '#5dade2'
                elif current_slope < 2:
                    color = '#2ecc71'
                elif current_slope < 5:
                    color = '#f39c12'
                else:
                    color = '#e74c3c'
                
                folium.PolyLine(
                    segment_coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Slope: {current_slope:.1f}%"
                ).add_to(slope_group)
    
    slope_group.add_to(m)
    
    # ========== TRAFFIC LAYERS ==========
    print("  Loading traffic data...")
    traffic_data = load_traffic_data()
    
    if traffic_data:
        print(f"  ✓ Loaded traffic data for {len(traffic_data)} waypoints")
        
        # Layer 5: Traffic Overview (Average)
        add_traffic_layer(m, traffic_data, 'Traffic - Overview (Avg)', hour=None)
        
        # Layer 6: Morning Peak (9am)
        add_traffic_layer(m, traffic_data, 'Traffic - Morning Peak (09:00)', hour=9)
        
        # Layer 7: Evening Peak (6pm)
        add_traffic_layer(m, traffic_data, 'Traffic - Evening Peak (18:00)', hour=18)
        
        # Layer 8: Off-Peak (9pm)
        add_traffic_layer(m, traffic_data, 'Traffic - Off-Peak (21:00)', hour=21)
        
        # Layer 9: Early Morning (6am)
        add_traffic_layer(m, traffic_data, 'Traffic - Early Morning (06:00)', hour=6)
        
        print("  ✓ Added 5 traffic layers")
    else:
        print("  ⚠ No traffic data found")
    
    # Add waypoint markers
    waypoint_group = folium.FeatureGroup(name='Waypoints', show=True)
    
    summary_file = MIDPOINTS_DIR / "all_midpoints_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            midpoints_data = json.load(f)
        
        for route_data in midpoints_data:
            # Start marker
            start = route_data['start']
            folium.Marker(
                [start['lat'], start['lon']],
                popup=f"START: {route_data['source']}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(waypoint_group)
            
            # End marker
            end = route_data['end']
            folium.Marker(
                [end['lat'], end['lon']],
                popup=f"END: {route_data['destination']}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(waypoint_group)
    
    waypoint_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add comprehensive legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 280px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                padding: 10px; max-height: 500px; overflow-y: auto;">
    <p><b>LEGEND</b></p>
    
    <p><b>Speed Limits:</b></p>
    <p><span style="color:#e74c3c">━━━</span> ≤50 km/h</p>
    <p><span style="color:#f39c12">━━━</span> 51-70 km/h</p>
    <p><span style="color:#f1c40f">━━━</span> 71-90 km/h</p>
    <p><span style="color:#3498db">━━━</span> 91-110 km/h</p>
    <p><span style="color:#2ecc71">━━━</span> >110 km/h</p>
    
    <p><b>Road Classes:</b></p>
    <p><span style="color:#e74c3c">━━━</span> Motorway</p>
    <p><span style="color:#e67e22">━━━</span> Trunk</p>
    <p><span style="color:#f39c12">━━━</span> Primary</p>
    <p><span style="color:#f1c40f">━━━</span> Secondary</p>
    <p><span style="color:#3498db">━━━</span> Tertiary</p>
    
    <p><b>Slopes:</b></p>
    <p><span style="color:#3498db">━━━</span> Steep down (<-5%)</p>
    <p><span style="color:#5dade2">━━━</span> Down (-2 to -5%)</p>
    <p><span style="color:#2ecc71">━━━</span> Flat (±2%)</p>
    <p><span style="color:#f39c12">━━━</span> Up (2-5%)</p>
    <p><span style="color:#e74c3c">━━━</span> Steep up (>5%)</p>
    
    <p><b>Traffic Congestion:</b></p>
    <p><i class="fa fa-circle" style="color:#2ecc71"></i> Free Flow (0-20%)</p>
    <p><i class="fa fa-circle" style="color:#f1c40f"></i> Moderate (20-40%)</p>
    <p><i class="fa fa-circle" style="color:#e67e22"></i> Heavy (40-60%)</p>
    <p><i class="fa fa-circle" style="color:#e74c3c"></i> Congested (60%+)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_file = MAPS_DIR / "combined_features_map.html"
    m.save(str(output_file))
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"\n✓ Combined map saved: {output_file.name} ({file_size:.1f} MB)")
    print(f"  Layers: Base Routes, Speed Limits, Road Classes, Slopes, Waypoints")
    print(f"  Traffic: Overview, Morning Peak, Evening Peak, Off-Peak, Early Morning")
    
    return output_file


def main():
    print("=" * 80)
    print("Combined Features + Traffic Map Generator")
    print("=" * 80)
    
    create_combined_map_with_traffic()
    
    print("\n" + "=" * 80)
    print("Map Generation Complete!")
    print("=" * 80)
    print("\n✓ Open combined_features_map.html in a browser to view")
    print("  Use the layer control in the top-right corner to toggle features")


if __name__ == "__main__":
    main()
