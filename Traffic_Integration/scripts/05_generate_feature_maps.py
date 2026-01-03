#!/usr/bin/env python3
"""
Generate enhanced visualization maps showing route features and speed limits.
Fixed version with continuous line rendering (no gaps).
"""

import json
import folium
from pathlib import Path

def create_speed_limit_map(selected_routes_dir, features_dir, output_file):
    """Create map colored by speed limits - continuous lines."""
    
    print("\n" + "="*80)
    print("GENERATING SPEED LIMIT MAP")
    print("="*80)
    
    belgium_center = [50.8503, 4.3517]
    m = folium.Map(location=belgium_center, zoom_start=9, tiles='OpenStreetMap')
    
    speed_colors = {120: '#c0392b', 90: '#e74c3c', 70: '#f39c12', 
                   50: '#f1c40f', 30: '#2ecc71', None: '#95a5a6'}
    
    feature_files = sorted(Path(features_dir).glob('*_features.json'))
    
    for feature_file in feature_files:
        with open(feature_file, 'r') as f:
            features = json.load(f)
        
        route_name = f"{features['route_info']['source']} → {features['route_info']['destination']}"
        print(f"\nProcessing: {route_name}")
        
        route_file = Path(selected_routes_dir) / f"{feature_file.stem.replace('_features', '')}.json"
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        coordinates = route_data['routes'][0]['path']['points']['coordinates']
        
        # Map all indices to speeds
        speed_map = {}
        for segment in features['max_speed']:
            for i in range(segment['start_index'], segment['end_index']):
                speed_map[i] = segment['value']
        
        # Draw continuous line
        i = 0
        while i < len(coordinates) - 1:
            speed = speed_map.get(i, None)
            color = speed_colors.get(speed, speed_colors[None])
            
            j = i + 1
            while j < len(coordinates) and speed_map.get(j, None) == speed:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            speed_text = f"{speed:.0f} km/h" if speed else "Unknown"
            
            folium.PolyLine(seg_coords, color=color, weight=5, opacity=0.8,
                          popup=f"<b>{route_name}</b><br>Speed: {speed_text}").add_to(m)
            i = j
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 280px; 
                background-color: white; border:3px solid #2c3e50; z-index:9999; 
                font-size:13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="margin-top:0; color: #2c3e50;">Speed Limits</h4>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="margin: 4px 0;"><span style="color: #c0392b; font-size: 20px;">━━━</span> 120 km/h</p>
        <p style="margin: 4px 0;"><span style="color: #e74c3c; font-size: 20px;">━━━</span> 90 km/h</p>
        <p style="margin: 4px 0;"><span style="color: #f39c12; font-size: 20px;">━━━</span> 70 km/h</p>
        <p style="margin: 4px 0;"><span style="color: #f1c40f; font-size: 20px;">━━━</span> 50 km/h</p>
        <p style="margin: 4px 0;"><span style="color: #2ecc71; font-size: 20px;">━━━</span> 30 km/h</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_file)
    print(f"\n✓ Saved: {output_file}")

def create_road_class_map(selected_routes_dir, features_dir, output_file):
    """Create map colored by road class - continuous lines."""
    
    print("\n" + "="*80)
    print("GENERATING ROAD CLASS MAP")
    print("="*80)
    
    belgium_center = [50.8503, 4.3517]
    m = folium.Map(location=belgium_center, zoom_start=9, tiles='OpenStreetMap')
    
    class_colors = {'motorway': '#8e44ad', 'trunk': '#c0392b', 'primary': '#e74c3c',
                   'secondary': '#f39c12', 'tertiary': '#f1c40f', 'residential': '#2ecc71',
                   'service': '#95a5a6', 'unclassified': '#bdc3c7', 'track': '#7f8c8d'}
    
    feature_files = sorted(Path(features_dir).glob('*_features.json'))
    
    for feature_file in feature_files:
        with open(feature_file, 'r') as f:
            features = json.load(f)
        
        route_name = f"{features['route_info']['source']} → {features['route_info']['destination']}"
        print(f"\nProcessing: {route_name}")
        
        route_file = Path(selected_routes_dir) / f"{feature_file.stem.replace('_features', '')}.json"
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        coordinates = route_data['routes'][0]['path']['points']['coordinates']
        
        # Map all indices to road classes
        class_map = {}
        for segment in features['road_class']:
            for i in range(segment['start_index'], segment['end_index']):
                class_map[i] = segment['value']
        
        # Draw continuous line
        i = 0
        while i < len(coordinates) - 1:
            road_class = class_map.get(i, 'unknown')
            color = class_colors.get(road_class, '#34495e')
            
            j = i + 1
            while j < len(coordinates) and class_map.get(j, 'unknown') == road_class:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            
            folium.PolyLine(seg_coords, color=color, weight=5, opacity=0.8,
                          popup=f"<b>{route_name}</b><br>Class: {road_class}").add_to(m)
            i = j
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 280px; 
                background-color: white; border:3px solid #2c3e50; z-index:9999; 
                font-size:13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="margin-top:0; color: #2c3e50;">Road Classifications</h4>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="margin: 4px 0;"><span style="color: #8e44ad; font-size: 20px;">━━━</span> Motorway</p>
        <p style="margin: 4px 0;"><span style="color: #c0392b; font-size: 20px;">━━━</span> Trunk</p>
        <p style="margin: 4px 0;"><span style="color: #e74c3c; font-size: 20px;">━━━</span> Primary</p>
        <p style="margin: 4px 0;"><span style="color: #f39c12; font-size: 20px;">━━━</span> Secondary</p>
        <p style="margin: 4px 0;"><span style="color: #f1c40f; font-size: 20px;">━━━</span> Tertiary</p>
        <p style="margin: 4px 0;"><span style="color: #2ecc71; font-size: 20px;">━━━</span> Residential</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_file)
    print(f"\n✓ Saved: {output_file}")

def create_slope_map(selected_routes_dir, features_dir, output_file):
    """Create map colored by slope - continuous lines."""
    
    print("\n" + "="*80)
    print("GENERATING SLOPE MAP")
    print("="*80)
    
    belgium_center = [50.8503, 4.3517]
    m = folium.Map(location=belgium_center, zoom_start=9, tiles='OpenStreetMap')
    
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
    
    feature_files = sorted(Path(features_dir).glob('*_features.json'))
    
    for feature_file in feature_files:
        with open(feature_file, 'r') as f:
            features = json.load(f)
        
        route_name = f"{features['route_info']['source']} → {features['route_info']['destination']}"
        print(f"\nProcessing: {route_name}")
        
        route_file = Path(selected_routes_dir) / f"{feature_file.stem.replace('_features', '')}.json"
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        coordinates = route_data['routes'][0]['path']['points']['coordinates']
        
        # Map all indices to slopes
        slope_map = {}
        for segment in features['average_slope']:
            for i in range(segment['start_index'], segment['end_index']):
                slope_map[i] = segment['value']
        
        # Draw continuous line
        i = 0
        while i < len(coordinates) - 1:
            slope = slope_map.get(i, 0)
            color = get_slope_color(slope)
            
            j = i + 1
            while j < len(coordinates) and slope_map.get(j, 0) == slope:
                j += 1
            
            seg_coords = [[coordinates[k][1], coordinates[k][0]] for k in range(i, min(j + 1, len(coordinates)))]
            slope_text = f"{slope:.1f}%"
            
            folium.PolyLine(seg_coords, color=color, weight=5, opacity=0.8,
                          popup=f"<b>{route_name}</b><br>Slope: {slope_text}").add_to(m)
            i = j
    
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 280px; 
                background-color: white; border:3px solid #2c3e50; z-index:9999; 
                font-size:13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="margin-top:0; color: #2c3e50;">Road Slopes</h4>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="margin: 4px 0;"><span style="color: #c0392b; font-size: 20px;">━━━</span> Steep Uphill (&gt;5%)</p>
        <p style="margin: 4px 0;"><span style="color: #e74c3c; font-size: 20px;">━━━</span> Uphill (2-5%)</p>
        <p style="margin: 4px 0;"><span style="color: #3498db; font-size: 20px;">━━━</span> Flat (±2%)</p>
        <p style="margin: 4px 0;"><span style="color: #27ae60; font-size: 20px;">━━━</span> Downhill (-2 to -5%)</p>
        <p style="margin: 4px 0;"><span style="color: #16a085; font-size: 20px;">━━━</span> Steep Downhill (&lt;-5%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_file)
    print(f"\n✓ Saved: {output_file}")

def main():
    selected_routes_dir = "../selected_routes"
    features_dir = "../route_features"
    output_dir = "../maps"
    
    print("="*80)
    print("GENERATING FEATURE VISUALIZATION MAPS (NO GAPS)")
    print("="*80)
    
    create_speed_limit_map(selected_routes_dir, features_dir, f"{output_dir}/speed_limits_map.html")
    create_road_class_map(selected_routes_dir, features_dir, f"{output_dir}/road_classes_map.html")
    create_slope_map(selected_routes_dir, features_dir, f"{output_dir}/slopes_map.html")
    
    print("\n" + "="*80)
    print("✓ ALL FEATURE MAPS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated maps in: {output_dir}/")
    print("  - speed_limits_map.html")
    print("  - road_classes_map.html")
    print("  - slopes_map.html")
    print("="*80)

if __name__ == "__main__":
    main()
