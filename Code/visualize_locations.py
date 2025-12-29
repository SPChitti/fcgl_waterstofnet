"""
FCGL Location Visualizer
Reads master_locations.csv and generates an interactive map of Belgium with all locations.
"""
import pandas as pd
import folium
from pathlib import Path


def load_locations(csv_path):
    """Load location data from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} locations")
    return df


def create_belgium_map(locations_df, output_path):
    """
    Create an interactive map of Belgium with all locations marked.
    
    Args:
        locations_df: DataFrame with columns [Location, Type, Latitude, Longitude]
        output_path: Path where the HTML map will be saved
    """
    # Center map on Belgium (approximate center of Flanders)
    belgium_center = [50.9, 4.0]
    
    # Create base map
    m = folium.Map(
        location=belgium_center,
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Define color coding based on location type
    def get_marker_color(loc_type):
        if 'port' in loc_type.lower() or 'h₂' in loc_type.lower():
            return 'blue'
        elif 'hub' in loc_type.lower():
            return 'red'
        elif 'industrial' in loc_type.lower():
            return 'darkred'
        elif 'logistics' in loc_type.lower():
            return 'orange'
        else:
            return 'green'
    
    # Add markers for each location
    for idx, row in locations_df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"<b>{row['Location']}</b><br>{row['Type']}",
            tooltip=row['Location'],
            icon=folium.Icon(
                color=get_marker_color(row['Type']),
                icon='info-sign'
            )
        ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin:0; font-weight:bold;">Location Types</p>
    <p style="margin:5px 0;"><i class="fa fa-map-marker" style="color:blue"></i> Major Port / H₂</p>
    <p style="margin:5px 0;"><i class="fa fa-map-marker" style="color:red"></i> Hub</p>
    <p style="margin:5px 0;"><i class="fa fa-map-marker" style="color:darkred"></i> Industrial</p>
    <p style="margin:5px 0;"><i class="fa fa-map-marker" style="color:orange"></i> Logistics</p>
    <p style="margin:5px 0;"><i class="fa fa-map-marker" style="color:green"></i> Other</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(str(output_path))
    print(f"Map saved to: {output_path}")
    
    return m


def main():
    """Main execution function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'Data'
    maps_dir = project_root / 'Maps'
    
    # Create maps directory if it doesn't exist
    maps_dir.mkdir(exist_ok=True)
    
    # Load location data
    csv_path = data_dir / 'master_locations.csv'
    locations_df = load_locations(csv_path)
    
    # Create and save map
    output_path = maps_dir / 'belgium_locations_map.html'
    create_belgium_map(locations_df, output_path)
    
    print("\n" + "="*50)
    print("✓ Visualization complete!")
    print(f"✓ Open {output_path} in your browser to view the map")
    print("="*50)


if __name__ == "__main__":
    main()
