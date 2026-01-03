#!/usr/bin/env python3
"""
Generate combined map with all route features in separate layers.
Uses Folium layer control to toggle between speed limits, road classes, 
slopes, and base routes.
"""

import json
import folium
from pathlib import Path

def create_combined_map(selected_routes_dir, features_dir, midpoints_dir, output_file):
    """Create single map with all features as toggleable layers."""
    
    print("="*80)
    print("GENERATING COMBINED FEATURE MAP")
    print("="*80)
    
    belgium_center = [50.8503, 4.3517]
    m = folium.Map(location=belgium_center, zoom_start=9, tiles='OpenStreetMap')
    
    # Color schemes
    speed_colors = {120: '#c0392b', 90: '#e74c3c', 70: '#f39c12', 
                   50: '#f1c40f', 30: '#2ecc71', None: '#95a5a6'}
    
    class_colors = {'motorway': '#8e44ad', 'trunk': '#c0392b', 'primary': '#e74c3c',
                   'secondary': '#f39c12', 'tertiary': '#f1c40f', 'residential': '#2ecc71',
                   'service': '#95a5a6', 'unclassified': '#bdc3c7', 'track': '#7f8c8d'}
    
    def get_slope_color(slope):
        if slope is None or slope == 0:
            return '#3498db'
        elif slope > 5:
            return '#c0392b'
        elif slope > 2:
            return '#e74c3c'
        elif slope > -2:
            return '#3498db'
        elif slope > -5:
            return '#27ae60'
        else:
            return '#16a085'
    
    # Route colors for base layer
    route_colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # Create feature groups for layer control
    base_layer = folium.FeatureGroup(name='Base Routes with Waypoints', show=True)
    speed_layer = folium.FeatureGroup(name='Speed Limits', show=False)
    class_layer = folium.FeatureGroup(name='Road Classes', show=False)
    slope_layer = folium.FeatureGroup(name='Slopes', show=False)
    
    # Load all routes
    feature_files = sorted(Path(features_dir).glob('*_features.json'))
    
    for idx, feature_file in enumerate(feature_files):
        print(f"\nProcessing Route {idx + 1}...")
        
        with open(feature_file, 'r') as f:
            features = json.load(f)
        
        route_name = f"{features['route_info']['source']} → {features['route_info']['destination']}"
        print(f"  {route_name}")
        
        # Load full route
        route_file = Path(selected_routes_dir) / f"{feature_file.stem.replace('_features', '')}.json"
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        coordinates = route_data['routes'][0]['path']['points']['coordinates']
        
        # 1. BASE LAYER - Simple route line with waypoints
        route_coords = [[coord[1], coord[0]] for coord in coordinates]
        folium.PolyLine(
            route_coords,
            color=route_colors[idx % len(route_colors)],
            weight=4,
            opacity=0.7,
            popup=f"<b>{route_name}</b><br>Distance: {features['route_info']['total_distance_km']:.1f} km"
        ).add_to(base_layer)
        
        # Add waypoints to base layer
        midpoint_file = Path(midpoints_dir) / f"{feature_file.stem.replace('_features', '')}_midpoints.json"
        if midpoint_file.exists():
            with open(midpoint_file, 'r') as f:
                midpoint_data = json.load(f)
            
            for waypoint in midpoint_data['all_waypoints']:
                if waypoint['label'] == 'START':
                    icon_color, icon = 'green', 'play'
                elif waypoint['label'] == 'END':
                    icon_color, icon = 'red', 'stop'
                else:
                    icon_color, icon = 'blue', 'map-pin'
                
                folium.Marker(
                    location=[waypoint['lat'], waypoint['lon']],
                    popup=f"<b>{waypoint['label']}</b><br>{route_name}<br>{waypoint['distance_from_start_km']:.1f} km",
                    tooltip=f"{waypoint['label']}",
                    icon=folium.Icon(color=icon_color, icon=icon, prefix='fa')
                ).add_to(base_layer)
        
        # 2. SPEED LIMIT LAYER
        speed_map = {}
        for segment in features['max_speed']:
            for i in range(segment['start_index'], segment['end_index']):
                speed_map[i] = segment['value']
        
        i = 0
        while i < len(coordinates) - 1:
            speed = speed_map.get(i, None)
            color = speed_colors.get(speed, speed_colors[None])
            
            j = i + 1
            while j < len(coordinates) and speed_map.get(j, None) == speed:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            speed_text = f"{speed:.0f} km/h" if speed else "Unknown"
            
            folium.PolyLine(
                seg_coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"<b>{route_name}</b><br>Speed Limit: {speed_text}"
            ).add_to(speed_layer)
            i = j
        
        # 3. ROAD CLASS LAYER
        class_map = {}
        for segment in features['road_class']:
            for i in range(segment['start_index'], segment['end_index']):
                class_map[i] = segment['value']
        
        i = 0
        while i < len(coordinates) - 1:
            road_class = class_map.get(i, 'unknown')
            color = class_colors.get(road_class, '#34495e')
            
            j = i + 1
            while j < len(coordinates) and class_map.get(j, 'unknown') == road_class:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            
            folium.PolyLine(
                seg_coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"<b>{route_name}</b><br>Road Class: {road_class}"
            ).add_to(class_layer)
            i = j
        
        # 4. SLOPE LAYER
        slope_map = {}
        for segment in features['average_slope']:
            for i in range(segment['start_index'], segment['end_index']):
                slope_map[i] = segment['value']
        
        i = 0
        while i < len(coordinates) - 1:
            slope = slope_map.get(i, 0)
            color = get_slope_color(slope)
            
            j = i + 1
            while j < len(coordinates) and slope_map.get(j, 0) == slope:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            slope_text = f"{slope:.1f}%"
            
            folium.PolyLine(
                seg_coords,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"<b>{route_name}</b><br>Slope: {slope_text}"
            ).add_to(slope_layer)
            i = j
    
    # Add all layers to map
    base_layer.add_to(m)
    speed_layer.add_to(m)
    class_layer.add_to(m)
    slope_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add comprehensive legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 320px; max-height: 500px;
                overflow-y: auto; background-color: white; border:3px solid #2c3e50; z-index:9999; 
                font-size:12px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="margin-top:0; color: #2c3e50;">Combined Route Features</h4>
        <p style="font-size: 11px; color: #7f8c8d; margin: 5px 0;">
            Use layer control (top-right) to toggle features
        </p>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        
        <p style="font-weight: bold; color: #2c3e50; margin: 8px 0 4px 0;">Speed Limits:</p>
        <p style="margin: 2px 0;"><span style="color: #c0392b; font-size: 16px;">━━</span> 120 km/h</p>
        <p style="margin: 2px 0;"><span style="color: #e74c3c; font-size: 16px;">━━</span> 90 km/h</p>
        <p style="margin: 2px 0;"><span style="color: #f39c12; font-size: 16px;">━━</span> 70 km/h</p>
        <p style="margin: 2px 0;"><span style="color: #f1c40f; font-size: 16px;">━━</span> 50 km/h</p>
        <p style="margin: 2px 0;"><span style="color: #2ecc71; font-size: 16px;">━━</span> 30 km/h</p>
        
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="font-weight: bold; color: #2c3e50; margin: 8px 0 4px 0;">Road Classes:</p>
        <p style="margin: 2px 0;"><span style="color: #8e44ad; font-size: 16px;">━━</span> Motorway</p>
        <p style="margin: 2px 0;"><span style="color: #e74c3c; font-size: 16px;">━━</span> Primary</p>
        <p style="margin: 2px 0;"><span style="color: #f39c12; font-size: 16px;">━━</span> Secondary</p>
        <p style="margin: 2px 0;"><span style="color: #2ecc71; font-size: 16px;">━━</span> Residential</p>
        
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="font-weight: bold; color: #2c3e50; margin: 8px 0 4px 0;">Slopes:</p>
        <p style="margin: 2px 0;"><span style="color: #c0392b; font-size: 16px;">━━</span> Steep Up (&gt;5%)</p>
        <p style="margin: 2px 0;"><span style="color: #e74c3c; font-size: 16px;">━━</span> Uphill (2-5%)</p>
        <p style="margin: 2px 0;"><span style="color: #3498db; font-size: 16px;">━━</span> Flat (±2%)</p>
        <p style="margin: 2px 0;"><span style="color: #27ae60; font-size: 16px;">━━</span> Downhill (-2 to -5%)</p>
        
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="font-weight: bold; color: #2c3e50; margin: 8px 0 4px 0;">Routes:</p>
        <p style="margin: 2px 0;"><span style="color: #e74c3c; font-size: 16px;">━━</span> Dendermonde → Mechelen</p>
        <p style="margin: 2px 0;"><span style="color: #3498db; font-size: 16px;">━━</span> Genk → Aalst</p>
        <p style="margin: 2px 0;"><span style="color: #2ecc71; font-size: 16px;">━━</span> Waregem → Sint-Niklaas</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save(output_file)
    
    print("\n" + "="*80)
    print(f"✓ Combined map saved: {output_file}")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Base Routes with Waypoints (default on)")
    print("  ✓ Speed Limits layer")
    print("  ✓ Road Classes layer")
    print("  ✓ Slopes layer")
    print("\nUse the layer control in top-right corner to toggle features!")
    print("="*80)

def main():
    selected_routes_dir = "../selected_routes"
    features_dir = "../route_features"
    midpoints_dir = "../midway_points"
    output_file = "../maps/combined_features_map.html"
    
    create_combined_map(selected_routes_dir, features_dir, midpoints_dir, output_file)

if __name__ == "__main__":
    main()
