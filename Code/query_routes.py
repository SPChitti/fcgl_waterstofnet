"""
Query GraphHopper for routes between Belgium logistics locations.
Extracts routes with full attribution (geometry, slope, road class, speed limits, etc.)
for FCGL training data preparation.
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# GraphHopper server URL
GRAPHHOPPER_URL = "http://localhost:8080"

# Vehicle profiles
PROFILES = ["truck_diesel", "truck_ev", "truck_h2"]

class GraphHopperClient:
    """Client for querying GraphHopper routing API."""
    
    def __init__(self, base_url: str = GRAPHHOPPER_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if GraphHopper server is running."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        profile: str = "truck_diesel",
        details: Optional[List[str]] = None
    ) -> Dict:
        """
        Query route between two points.
        
        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple
            profile: Vehicle profile name
            details: List of edge attributes to include (e.g., 'road_class', 'max_slope')
        
        Returns:
            Route response dict with paths, geometry, and details
        """
        if details is None:
            details = [
                "road_class",
                "road_access", 
                "max_slope",
                "average_slope",
                "max_speed",
                "surface",
                "road_environment",
                "distance",
                "time"
            ]
        
        params = {
            "profile": profile,
            "point": [f"{origin[0]},{origin[1]}", f"{destination[0]},{destination[1]}"],
            "points_encoded": False,  # Get human-readable coordinates
            "instructions": True,
            "details": ",".join(details),
            "elevation": True,
            "calc_points": True
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/route",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying route: {e}")
            return None
    
    def extract_route_features(self, route_response: Dict) -> Dict:
        """
        Extract relevant features from route response for FCGL.
        
        Returns:
            Dict with route metadata and edge-level features
        """
        if not route_response or "paths" not in route_response:
            return None
        
        path = route_response["paths"][0]
        
        features = {
            "distance_m": path.get("distance", 0),
            "time_ms": path.get("time", 0),
            "ascend_m": path.get("ascend", 0),
            "descend_m": path.get("descend", 0),
            "points": path.get("points", {}).get("coordinates", []),
            "details": path.get("details", {}),
            "instructions": path.get("instructions", [])
        }
        
        return features


def load_locations(csv_path: str = "../Data/master_locations.csv") -> pd.DataFrame:
    """Load location data from CSV."""
    df = pd.read_csv(csv_path)
    return df


def generate_od_pairs(
    locations: pd.DataFrame,
    max_pairs: Optional[int] = None
) -> List[Tuple[Dict, Dict]]:
    """
    Generate origin-destination pairs from locations.
    
    Args:
        locations: DataFrame with location data
        max_pairs: Maximum number of pairs to generate (None = all)
    
    Returns:
        List of (origin_dict, destination_dict) tuples
    """
    od_pairs = []
    
    for i, origin in locations.iterrows():
        for j, dest in locations.iterrows():
            if i != j:  # Skip same location
                od_pairs.append((
                    {
                        "id": origin["Location_ID"],
                        "name": origin["Location_Name"],
                        "lat": origin["Latitude"],
                        "lon": origin["Longitude"],
                        "type": origin["Type"]
                    },
                    {
                        "id": dest["Location_ID"],
                        "name": dest["Location_Name"],
                        "lat": dest["Latitude"],
                        "lon": dest["Longitude"],
                        "type": dest["Type"]
                    }
                ))
    
    if max_pairs:
        od_pairs = od_pairs[:max_pairs]
    
    return od_pairs


def query_and_save_routes(
    client: GraphHopperClient,
    od_pairs: List[Tuple[Dict, Dict]],
    output_dir: str = "../Data/routes",
    profiles: List[str] = PROFILES
):
    """
    Query routes for all OD pairs and profiles, save to files.
    
    Args:
        client: GraphHopper client
        od_pairs: List of origin-destination pairs
        output_dir: Directory to save route data
        profiles: List of vehicle profiles to query
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_queries = len(od_pairs) * len(profiles)
    print(f"\nQuerying {total_queries} routes ({len(od_pairs)} OD pairs × {len(profiles)} profiles)")
    print("=" * 80)
    
    all_routes = []
    query_count = 0
    
    for origin, dest in od_pairs:
        for profile in profiles:
            query_count += 1
            print(f"\n[{query_count}/{total_queries}] {origin['name']} → {dest['name']} ({profile})")
            
            # Query route
            route_response = client.route(
                origin=(origin['lat'], origin['lon']),
                destination=(dest['lat'], dest['lon']),
                profile=profile
            )
            
            if route_response:
                features = client.extract_route_features(route_response)
                
                if features:
                    route_data = {
                        "origin_id": origin['id'],
                        "origin_name": origin['name'],
                        "origin_lat": origin['lat'],
                        "origin_lon": origin['lon'],
                        "origin_type": origin['type'],
                        "dest_id": dest['id'],
                        "dest_name": dest['name'],
                        "dest_lat": dest['lat'],
                        "dest_lon": dest['lon'],
                        "dest_type": dest['type'],
                        "profile": profile,
                        "distance_m": features['distance_m'],
                        "time_ms": features['time_ms'],
                        "ascend_m": features['ascend_m'],
                        "descend_m": features['descend_m'],
                        "num_points": len(features['points']),
                        "route_data": features
                    }
                    
                    all_routes.append(route_data)
                    
                    print(f"  ✓ Distance: {features['distance_m']/1000:.1f} km, "
                          f"Time: {features['time_ms']/60000:.1f} min, "
                          f"Points: {len(features['points'])}")
                else:
                    print(f"  ✗ Failed to extract features")
            else:
                print(f"  ✗ Route query failed")
            
            # Rate limiting
            time.sleep(0.1)
    
    # Save all routes to JSON
    output_file = output_path / "all_routes.json"
    with open(output_file, 'w') as f:
        json.dump(all_routes, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Saved {len(all_routes)} routes to {output_file}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            "origin": r['origin_name'],
            "destination": r['dest_name'],
            "profile": r['profile'],
            "distance_km": r['distance_m'] / 1000,
            "time_min": r['time_ms'] / 60000,
            "ascend_m": r['ascend_m'],
            "descend_m": r['descend_m']
        }
        for r in all_routes
    ])
    
    summary_file = output_path / "routes_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to {summary_file}")
    
    return all_routes


def main():
    """Main entry point."""
    print("=" * 80)
    print("GraphHopper Route Query Script for FCGL Belgium Logistics")
    print("=" * 80)
    
    # Initialize client
    client = GraphHopperClient()
    
    # Check server health
    print("\n1. Checking GraphHopper server...")
    if not client.health_check():
        print("✗ GraphHopper server not responding at", GRAPHHOPPER_URL)
        print("  Please start the server first: java -jar graphhopper-web-9.1.jar server config.yml")
        return
    print("✓ Server is running")
    
    # Load locations
    print("\n2. Loading locations...")
    locations = load_locations()
    print(f"✓ Loaded {len(locations)} locations from master_locations.csv")
    
    # Generate OD pairs
    print("\n3. Generating OD pairs...")
    od_pairs = generate_od_pairs(locations)
    print(f"✓ Generated {len(od_pairs)} OD pairs")
    
    # Query routes
    print("\n4. Querying routes from GraphHopper...")
    routes = query_and_save_routes(client, od_pairs)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Complete! Retrieved {len(routes)} routes.")
    print(f"  Next steps:")
    print(f"  - Review routes_summary.csv for distance/time distributions")
    print(f"  - Use all_routes.json for FCGL graph construction")
    print("=" * 80)


if __name__ == "__main__":
    main()
