#!/usr/bin/env python3
"""
Generate visualization map for the 3 selected routes with their midway points.
"""

import json
import folium
from pathlib import Path

def create_base_map(selected_routes_dir, midpoints_dir, output_file):
    """
    Create interactive map showing all 3 routes with their waypoints.
    """
    print("="*80)
    print("GENERATING BASE MAP")
    print("="*80)
    
    # Belgium center
    belgium_center = [50.8503, 4.3517]
    
    # Create base map
    m = folium.Map(
        location=belgium_center,
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Route colors
    route_colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    # Load midpoint files
    midpoint_files = sorted(Path(midpoints_dir).glob('*_midpoints.json'))
    midpoint_files = [f for f in midpoint_files if f.name != 'all_midpoints_summary.json']
    
    print(f"\nProcessing {len(midpoint_files)} routes...")
    
    for idx, midpoint_file in enumerate(midpoint_files):
        print(f"\nRoute {idx + 1}: {midpoint_file.stem}")
        
        # Load midpoint data
        with open(midpoint_file, 'r') as f:
            midpoint_data = json.load(f)
        
        # Load full route data
        route_file = Path(selected_routes_dir) / f"{midpoint_file.stem.replace('_midpoints', '')}.json"
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        
        route = route_data['routes'][0]
        coordinates = route['path']['points']['coordinates']
        
        color = route_colors[idx % len(route_colors)]
        
        # Draw route line
        route_coords = [[coord[1], coord[0]] for coord in coordinates]
        folium.PolyLine(
            route_coords,
            color=color,
            weight=4,
            opacity=0.7,
            popup=f"{midpoint_data['source']} → {midpoint_data['destination']}<br>"
                  f"Distance: {midpoint_data['total_distance_km']:.1f} km"
        ).add_to(m)
        
        # Add waypoint markers
        for waypoint in midpoint_data['all_waypoints']:
            if waypoint['label'] == 'START':
                icon_color = 'green'
                icon = 'play'
            elif waypoint['label'] == 'END':
                icon_color = 'red'
                icon = 'stop'
            else:
                icon_color = 'blue'
                icon = 'map-pin'
            
            folium.Marker(
                location=[waypoint['lat'], waypoint['lon']],
                popup=f"<b>{waypoint['label']}</b><br>"
                      f"Route: {midpoint_data['source']} → {midpoint_data['destination']}<br>"
                      f"Distance: {waypoint['distance_from_start_km']:.1f} km<br>"
                      f"Lat: {waypoint['lat']:.6f}<br>"
                      f"Lon: {waypoint['lon']:.6f}",
                tooltip=f"{waypoint['label']} ({waypoint['distance_from_start_km']:.1f} km)",
                icon=folium.Icon(color=icon_color, icon=icon, prefix='fa')
            ).add_to(m)
        
        print(f"  ✓ Added route with {len(midpoint_data['all_waypoints'])} waypoints")
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 320px; 
                background-color: white; border:3px solid #2c3e50; z-index:9999; 
                font-size:13px; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        <h4 style="margin-top:0; color: #2c3e50;">Selected Routes for Traffic Analysis</h4>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="margin: 4px 0;"><span style="color: #e74c3c; font-size: 18px;">━━━</span> 
           <b>Route 1:</b> Dendermonde → Mechelen (30.4 km)</p>
        <p style="margin: 4px 0;"><span style="color: #3498db; font-size: 18px;">━━━</span> 
           <b>Route 2:</b> Waregem → Sint-Niklaas (63.7 km)</p>
        <p style="margin: 4px 0;"><span style="color: #2ecc71; font-size: 18px;">━━━</span> 
           <b>Route 3:</b> Genk → Aalst (115.7 km)</p>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="margin: 8px 0; font-weight: bold; color: #2c3e50;">Waypoints:</p>
        <p style="margin: 4px 0;"><i class="fa fa-play" style="color: green;"></i> Start Point</p>
        <p style="margin: 4px 0;"><i class="fa fa-map-pin" style="color: blue;"></i> Midway Points (5 per route)</p>
        <p style="margin: 4px 0;"><i class="fa fa-stop" style="color: red;"></i> End Point</p>
        <hr style="border: 1px solid #ecf0f1; margin: 10px 0;">
        <p style="font-size: 11px; color: #7f8c8d; margin: 5px 0;">
            Total: 3 routes, 21 waypoints<br>
            Ready for TomTom Traffic API queries
        </p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    
    print(f"\n{'='*80}")
    print(f"✓ Base map generated successfully!")
    print(f"  Output: {output_file}")
    print(f"{'='*80}\n")

def main():
    selected_routes_dir = "../selected_routes"
    midpoints_dir = "../midway_points"
    output_file = "../maps/base_map.html"
    
    create_base_map(selected_routes_dir, midpoints_dir, output_file)
    
    print("You can open the map in your browser to view the routes and waypoints.")

if __name__ == "__main__":
    main()
