"""
Step 3: Identify Overlapping Route Sections

Analyzes the 6 routes to identify:
- Unique sections (only used by 1 OD pair)
- Shared sections (used by multiple OD pairs)

This optimizes waypoint placement for traffic data collection.
"""

import json
import folium
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
ROUTES_DIR = BASE_DIR / "routes"
MAPS_DIR = BASE_DIR / "maps"

MAPS_DIR.mkdir(exist_ok=True)

# Distance threshold for considering points "close enough" (in meters)
PROXIMITY_THRESHOLD = 50  # 50 meters


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def load_all_routes():
    """Load all route data"""
    routes = {}
    
    route_files = sorted(ROUTES_DIR.glob("S*_to_D*.json"))
    
    for route_file in route_files:
        with open(route_file, 'r') as f:
            data = json.load(f)
        
        od_pair = data['metadata']['od_pair']
        coords = data['paths'][0]['points']['coordinates']
        
        # Simplify: keep every Nth point to speed up analysis
        step = max(1, len(coords) // 200)  # Keep ~200 points per route
        simplified = coords[::step]
        
        routes[od_pair] = {
            'coords': simplified,
            'full_coords': coords,
            'metadata': data['metadata']
        }
    
    return routes


def find_overlapping_segments(routes):
    """Identify which segments are shared between routes"""
    
    print("Analyzing route overlaps...")
    print(f"  Proximity threshold: {PROXIMITY_THRESHOLD}m\n")
    
    # For each point in each route, find which other routes pass nearby
    overlaps = defaultdict(lambda: defaultdict(list))
    
    route_names = list(routes.keys())
    
    for i, route1 in enumerate(route_names):
        coords1 = routes[route1]['coords']
        
        for j, route2 in enumerate(route_names):
            if i >= j:  # Skip self and already compared pairs
                continue
            
            coords2 = routes[route2]['coords']
            
            # Find matching segments
            matches = []
            for idx1, (lon1, lat1) in enumerate(coords1):
                for idx2, (lon2, lat2) in enumerate(coords2):
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    if dist < PROXIMITY_THRESHOLD:
                        matches.append((idx1, idx2, dist))
            
            if matches:
                overlaps[route1][route2] = matches
                overlaps[route2][route1] = matches
                
                print(f"  {route1} ↔ {route2}: {len(matches)} overlapping points")
    
    return overlaps


def classify_route_sections(routes, overlaps):
    """Classify each route segment as unique or shared"""
    
    sections = {}
    
    for route_name, route_data in routes.items():
        coords = route_data['coords']
        classification = []
        
        # Check each point
        for idx, coord in enumerate(coords):
            # Count how many other routes overlap at this point
            overlap_count = 0
            overlapping_routes = []
            
            for other_route in overlaps.get(route_name, {}):
                matches = overlaps[route_name][other_route]
                if any(m[0] == idx for m in matches):
                    overlap_count += 1
                    overlapping_routes.append(other_route)
            
            classification.append({
                'coord': coord,
                'overlap_count': overlap_count,
                'overlapping_with': overlapping_routes
            })
        
        sections[route_name] = classification
    
    return sections


def create_overlap_map(routes, sections):
    """Create map showing unique vs shared sections"""
    
    m = folium.Map(
        location=[50.85, 4.35],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Title
    title_html = """
    <div style="position: fixed; top: 10px; left: 50px; width: 400px; height: 60px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:16px;
                padding: 10px; text-align: center;">
    <b>Route Overlap Analysis</b><br>
    <small>Identifying unique vs shared sections</small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Color map for overlap levels
    route_colors = {
        'S1_to_D1': '#e74c3c',
        'S1_to_D2': '#3498db',
        'S1_to_D3': '#2ecc71',
        'S2_to_D1': '#f39c12',
        'S2_to_D2': '#9b59b6',
        'S2_to_D3': '#1abc9c'
    }
    
    # Create feature groups
    unique_group = folium.FeatureGroup(name='Unique Sections', show=True)
    shared_2_group = folium.FeatureGroup(name='Shared by 2 routes', show=True)
    shared_3_group = folium.FeatureGroup(name='Shared by 3+ routes', show=True)
    
    # Plot routes colored by overlap level
    for route_name, section_data in sections.items():
        color = route_colors[route_name]
        
        # Group consecutive points by overlap count
        current_overlap = None
        segment_coords = []
        
        for point in section_data:
            overlap = point['overlap_count']
            coord = point['coord']
            
            if overlap != current_overlap:
                # Save previous segment
                if segment_coords:
                    plot_segment(segment_coords, current_overlap, route_name, 
                               unique_group, shared_2_group, shared_3_group, color)
                
                segment_coords = [[coord[1], coord[0]]]
                current_overlap = overlap
            else:
                segment_coords.append([coord[1], coord[0]])
        
        # Plot final segment
        if segment_coords:
            plot_segment(segment_coords, current_overlap, route_name,
                       unique_group, shared_2_group, shared_3_group, color)
    
    # Add groups to map
    unique_group.add_to(m)
    shared_2_group.add_to(m)
    shared_3_group.add_to(m)
    
    # Layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
                padding: 10px;">
    <p><b>LEGEND</b></p>
    <p><b>Section Types:</b></p>
    <p><span style="color:#333; font-weight:bold">━━━</span> Unique (1 route only)</p>
    <p><span style="color:#666; font-weight:bold">━━━</span> Shared by 2 routes</p>
    <p><span style="color:#000; font-weight:bold">━━━</span> Shared by 3+ routes</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    output_file = MAPS_DIR / "route_overlaps.html"
    m.save(str(output_file))
    
    file_size = output_file.stat().st_size / 1024
    print(f"\n✓ Map saved: {output_file.name} ({file_size:.1f} KB)")


def plot_segment(coords, overlap_count, route_name, unique_group, shared_2_group, shared_3_group, base_color):
    """Plot a segment based on overlap level"""
    
    if overlap_count == 0:
        # Unique section - use route color
        folium.PolyLine(
            coords,
            color=base_color,
            weight=4,
            opacity=0.7,
            popup=f"{route_name}<br>Unique section"
        ).add_to(unique_group)
    elif overlap_count == 1:
        # Shared by 2 routes
        folium.PolyLine(
            coords,
            color='#555',
            weight=5,
            opacity=0.8,
            popup=f"{route_name}<br>Shared with 1 other route"
        ).add_to(shared_2_group)
    else:
        # Shared by 3+ routes
        folium.PolyLine(
            coords,
            color='#000',
            weight=6,
            opacity=0.9,
            popup=f"{route_name}<br>Shared with {overlap_count} other routes"
        ).add_to(shared_3_group)


def analyze_coverage(sections):
    """Calculate statistics about unique vs shared sections"""
    
    print("\n" + "=" * 80)
    print("SECTION ANALYSIS")
    print("=" * 80)
    
    total_points = 0
    unique_points = 0
    shared_2_points = 0
    shared_3plus_points = 0
    
    for route_name, section_data in sections.items():
        route_unique = sum(1 for p in section_data if p['overlap_count'] == 0)
        route_shared_2 = sum(1 for p in section_data if p['overlap_count'] == 1)
        route_shared_3 = sum(1 for p in section_data if p['overlap_count'] >= 2)
        route_total = len(section_data)
        
        total_points += route_total
        unique_points += route_unique
        shared_2_points += route_shared_2
        shared_3plus_points += route_shared_3
        
        print(f"\n{route_name}:")
        print(f"  Total points: {route_total}")
        print(f"  Unique: {route_unique} ({100*route_unique/route_total:.1f}%)")
        print(f"  Shared with 1: {route_shared_2} ({100*route_shared_2/route_total:.1f}%)")
        print(f"  Shared with 2+: {route_shared_3} ({100*route_shared_3/route_total:.1f}%)")
    
    print("\n" + "-" * 80)
    print("OVERALL:")
    print(f"  Total points analyzed: {total_points}")
    print(f"  Unique sections: {unique_points} ({100*unique_points/total_points:.1f}%)")
    print(f"  Shared by 2: {shared_2_points} ({100*shared_2_points/total_points:.1f}%)")
    print(f"  Shared by 3+: {shared_3plus_points} ({100*shared_3plus_points/total_points:.1f}%)")
    
    # Estimate unique edges needed
    print("\n" + "-" * 80)
    print("WAYPOINT ESTIMATION:")
    
    # Rough estimate: unique sections need their own waypoints
    # Shared sections need fewer waypoints (just once for the shared segment)
    estimated_unique_length = unique_points + (shared_2_points / 2) + (shared_3plus_points / 3)
    waypoints_needed = int(estimated_unique_length / 10)  # 1 waypoint per ~10 points
    
    print(f"  Estimated unique edge coverage: ~{int(estimated_unique_length)} points")
    print(f"  Suggested waypoints: ~{waypoints_needed} waypoints")
    print(f"  API calls (21 per waypoint): ~{waypoints_needed * 21}")


def main():
    print("=" * 80)
    print("Step 3: Route Overlap Analysis")
    print("=" * 80)
    print()
    
    # Load routes
    routes = load_all_routes()
    print(f"Loaded {len(routes)} routes\n")
    
    # Find overlaps
    overlaps = find_overlapping_segments(routes)
    
    # Classify sections
    sections = classify_route_sections(routes, overlaps)
    
    # Create map
    create_overlap_map(routes, sections)
    
    # Analyze coverage
    analyze_coverage(sections)
    
    print("\n" + "=" * 80)
    print("✓ Open maps/route_overlaps.html to view overlap analysis")


if __name__ == "__main__":
    main()
