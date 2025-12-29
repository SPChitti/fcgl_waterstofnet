# Route Generation for FCGL

## Overview
`generate_route.py` generates detailed routing data for Belgium logistics with:
- Multiple alternative routes (3-5 per OD pair)
- Comprehensive edge attributes (elevation, slopes, road classes, surfaces, environments)
- Node markers every 10 miles for graph creation
- Interactive HTML maps with route visualization

## Usage

### Basic Usage
```bash
python generate_route.py <source> <destination>
```

### With Parameters
```bash
python generate_route.py <source> <destination> [profile] [num_alternatives]
```

**Parameters:**
- `source`: Location name from master_locations.csv or coordinates "lat,lon"
- `destination`: Location name or coordinates
- `profile`: (optional) GraphHopper profile, default: truck_diesel
  - Options: truck_diesel, truck_ev, truck_h2
- `num_alternatives`: (optional) Number of alternative routes, default: 3, max: 5

### Examples

**Using location names:**
```bash
# 3 alternatives with diesel truck
python generate_route.py Antwerp Ghent

# 5 alternatives with EV truck
python generate_route.py "Port of Zeebrugge" Hasselt truck_ev 5

# Using partial match
python generate_route.py Zeebrugge Kortrijk
```

**Using coordinates:**
```bash
# Direct coordinates
python generate_route.py 51.2194,4.4025 51.0543,3.7174
```

## Output Files

All outputs saved to `../Maps/` directory:

### JSON File
`<source>_to_<destination>_<timestamp>.json`

Contains:
- Source/destination names and coordinates
- Profile used
- For each alternative route:
  - Full path with coordinates and elevations
  - Analysis: distance, duration, elevation gain/loss, slope statistics
  - Road class distribution
  - Surface types
  - Environment types (tunnel, bridge, road)
  - Speed limits
  - **Nodes**: Every 10 miles with geocoded place names, coordinates, elevations

### HTML Map
`<source>_to_<destination>_<timestamp>.html`

Interactive Folium map showing:
- All alternative routes (color-coded)
- Node markers every 10 miles
  - Green start marker
  - Red end marker
  - Blue intermediate nodes
- Popups with route/node details
- Click on routes to see distance, duration, elevation

## Available Locations

From `master_locations.csv`:
- Port of Zeebrugge
- Antwerp
- Ghent
- Bruges
- Ostend
- Kortrijk
- Roeselare
- Hasselt
- Genk
- Leuven
- Mechelen
- Aalst
- Sint-Niklaas
- Turnhout
- Dendermonde
- Lokeren
- Waregem
- Tielt

## Route Data Structure

### Analysis Object
```json
{
  "distance_km": 57.84,
  "distance_miles": 35.94,
  "duration_min": 41.1,
  "elevation": {
    "ascent_m": 292.0,
    "descent_m": 295.0,
    "net_elevation_m": -3.0
  },
  "slopes": {
    "uphill_segments": 53,
    "downhill_segments": 56,
    "steep_segments_over_5pct": 18,
    "max_uphill_grade_pct": 31.0,
    "max_downhill_grade_pct": -31.0
  },
  "road_classes": {"motorway": 2, "primary": 1, ...},
  "surfaces": {"asphalt": 9, "concrete": 2, ...},
  "environments": {"road": 21, "tunnel": 4, "bridge": 17},
  "avg_speed_kmh": 57.9
}
```

### Node Object
```json
{
  "node_id": 0,
  "place_name": "E17, Landmolen, Temse, Sint-Niklaas...",
  "latitude": 51.159984,
  "longitude": 4.234049,
  "elevation_m": 15.0,
  "distance_from_start_km": 16.16
}
```

## Use Cases

### 1. Graph Creation
Extract nodes every 10 miles with real place names and coordinates:
```python
import json
data = json.load(open('route.json'))
nodes = data['routes'][0]['nodes']  # Get nodes from first alternative
```

### 2. Route Comparison
Compare alternatives for same OD pair:
```python
for route in data['routes']:
    print(f"Alt {route['alternative_id']}: "
          f"{route['analysis']['distance_km']} km, "
          f"{route['analysis']['duration_min']} min")
```

### 3. Elevation Analysis
Analyze terrain characteristics:
```python
analysis = route['analysis']
print(f"Total climb: {analysis['elevation']['ascent_m']} m")
print(f"Steep segments: {analysis['slopes']['steep_segments_over_5pct']}")
```

## Next Steps

- Profile comparison (diesel vs EV vs H2) - separate analysis
- Generate routes for all 18×18 location pairs
- Aggregate statistics across Belgium network
- Traffic time-of-day multipliers
