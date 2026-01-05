"""
Step 9: Create Multi-Truck Adjacency List

Converts flow network to FCGL format:
- Each edge expanded to 3 rows (one per truck type: small, medium, heavy)
- Truck-specific: capacity, cost, CO2, travel time
- Keeps all road metadata: distance, slopes, traffic, speed limits
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "graph_config"

# Truck configurations (based on real hydrogen truck specs)
TRUCK_CONFIGS = {
    'small': {
        'capacity_kg': 350,  # Small hydrogen truck
        'avg_speed_kmph': 60,
        'max_speed_kmph': 80,
        'cost_per_km': 1.2,  # € per km
        'co2_emission_per_km': 0.15,  # kg CO2 per km (considering hydrogen production)
        'speed_penalty_slope': 0.15  # 15% speed reduction per % slope
    },
    'medium': {
        'capacity_kg': 1000,  # Medium hydrogen truck
        'avg_speed_kmph': 70,
        'max_speed_kmph': 90,
        'cost_per_km': 1.8,
        'co2_emission_per_km': 0.25,
        'speed_penalty_slope': 0.20  # 20% speed reduction per % slope
    },
    'heavy': {
        'capacity_kg': 4000,  # Heavy hydrogen truck
        'avg_speed_kmph': 80,
        'max_speed_kmph': 100,
        'cost_per_km': 2.5,
        'co2_emission_per_km': 0.40,
        'speed_penalty_slope': 0.25  # 25% speed reduction per % slope
    }
}

def calculate_truck_metrics(edge, truck_type):
    """Calculate truck-specific metrics for an edge"""
    
    truck = TRUCK_CONFIGS[truck_type]
    
    # Get edge properties
    distance_km = edge['distance_km']
    avg_slope = edge['avg_slope_pct']
    max_slope = edge['max_slope_pct']
    road_speed_limit = edge['avg_speed_kmh']
    morning_congestion = edge['morning_congestion_pct']
    
    # Effective speed considering:
    # 1. Truck speed capability
    # 2. Road speed limit
    # 3. Slope (reduces speed)
    # 4. Morning congestion (reduces speed)
    
    # Base speed (minimum of truck capability and road limit)
    base_speed = min(truck['avg_speed_kmph'], road_speed_limit)
    
    # Slope penalty
    slope_reduction = avg_slope * truck['speed_penalty_slope']
    speed_after_slope = base_speed * (1 - slope_reduction / 100)
    
    # Congestion penalty
    congestion_reduction = morning_congestion / 100
    effective_speed = speed_after_slope * (1 - congestion_reduction)
    
    # Ensure minimum speed
    effective_speed = max(effective_speed, 20.0)  # Minimum 20 km/h
    
    # Calculate metrics
    travel_time_hours = distance_km / effective_speed
    cost = distance_km * truck['cost_per_km']
    co2_kg = distance_km * truck['co2_emission_per_km']
    
    # Slope penalty on cost (steep slopes increase fuel/energy)
    if max_slope > 3.0:
        cost *= (1 + max_slope / 100)
    
    return {
        'truck_type': truck_type,
        'capacity_kg': truck['capacity_kg'],
        'effective_speed_kmh': round(effective_speed, 2),
        'travel_time_hours': round(travel_time_hours, 4),
        'cost': round(cost, 2),
        'co2_kg': round(co2_kg, 2)
    }

def create_adjacency_list():
    """Create multi-truck adjacency list in FCGL format"""
    
    print("="*80)
    print("Step 9: Creating Multi-Truck Adjacency List")
    print("="*80)
    
    # Load flow network edges
    print("\nLoading flow network edges...")
    edges_df = pd.read_csv(CONFIG_DIR / 'flow_network_edges.csv')
    print(f"  Loaded: {len(edges_df)} unique edges")
    
    # Create adjacency list with all truck types
    print("\nExpanding edges for 3 truck types...")
    adjacency_data = []
    
    for _, edge in edges_df.iterrows():
        for truck_type in ['small', 'medium', 'heavy']:
            # Calculate truck-specific metrics
            truck_metrics = calculate_truck_metrics(edge, truck_type)
            
            # Create row
            row = {
                'from_node': int(edge['from_node']),
                'to_node': int(edge['to_node']),
                'truck_type': truck_type,
                'capacity_kg': truck_metrics['capacity_kg'],
                'cost': truck_metrics['cost'],
                'co2_kg': truck_metrics['co2_kg'],
                'travel_time_hours': truck_metrics['travel_time_hours'],
                'effective_speed_kmh': truck_metrics['effective_speed_kmh'],
                'distance_km': round(edge['distance_km'], 3),
                
                # Road metadata
                'road_class': edge['road_class'],
                'lanes': int(edge['lanes']),
                'speed_limit_kmh': edge['avg_speed_kmh'],
                'max_speed_kmh': edge['max_speed_kmh'],
                'avg_slope_pct': edge['avg_slope_pct'],
                'max_slope_pct': edge['max_slope_pct'],
                'morning_congestion_pct': edge['morning_congestion_pct'],
                'offpeak_congestion_pct': edge['offpeak_congestion_pct'],
                'morning_avg_speed_kmh': edge['morning_avg_speed_kmh'],
                'offpeak_avg_speed_kmh': edge['offpeak_avg_speed_kmh'],
            }
            
            adjacency_data.append(row)
    
    adjacency_df = pd.DataFrame(adjacency_data)
    
    print(f"  Created: {len(adjacency_df)} edges (137 × 3 truck types = 411)")
    
    # Save adjacency list
    adjacency_file = CONFIG_DIR / 'adjacency_list_multimodal.csv'
    adjacency_df.to_csv(adjacency_file, index=False)
    print(f"  ✓ Saved: {adjacency_file}")
    
    # ==================== STATISTICS ====================
    print("\n" + "="*80)
    print("ADJACENCY LIST SUMMARY")
    print("="*80)
    
    print(f"\nTotal rows: {len(adjacency_df)}")
    print(f"  Unique physical edges: {len(edges_df)}")
    print(f"  Truck types per edge: 3")
    
    print("\nBy Truck Type:")
    for truck_type in ['small', 'medium', 'heavy']:
        subset = adjacency_df[adjacency_df['truck_type'] == truck_type]
        print(f"\n  {truck_type.upper()}:")
        print(f"    Edges: {len(subset)}")
        print(f"    Avg capacity: {subset['capacity_kg'].mean():.0f} kg")
        print(f"    Avg cost: €{subset['cost'].mean():.2f}")
        print(f"    Avg CO2: {subset['co2_kg'].mean():.2f} kg")
        print(f"    Avg time: {subset['travel_time_hours'].mean():.3f} hours")
        print(f"    Avg speed: {subset['effective_speed_kmh'].mean():.1f} km/h")
    
    print("\nRoad Metadata Included:")
    print("  ✓ distance_km")
    print("  ✓ speed_limit_kmh, max_speed_kmh")
    print("  ✓ avg_slope_pct, max_slope_pct")
    print("  ✓ morning_congestion_pct, offpeak_congestion_pct")
    print("  ✓ road_class, lanes")
    print("  ✗ curves/curvature (not available from GraphHopper)")
    
    print("\n" + "="*80)
    print("✓ Multi-Truck Adjacency List Created")
    print("="*80)
    print(f"\nFile: {adjacency_file}")
    print(f"Format: FCGL-compatible with full road metadata")
    print(f"Ready for training with supply/demand from demand.csv")
    print("="*80)
    
    return adjacency_df

def main():
    adjacency_df = create_adjacency_list()
    
    # Show sample
    print("\nSample rows (first 6 - same edge with 3 truck types):")
    print(adjacency_df[['from_node', 'to_node', 'truck_type', 'capacity_kg', 
                        'cost', 'travel_time_hours', 'effective_speed_kmh', 
                        'distance_km', 'road_class']].head(6))

if __name__ == '__main__':
    main()
