# FCGL Waterstofnet - Belgium Logistics Routing

Flow-Constrained Generative Learning for Belgium hydrogen logistics network.

## Overview

This project generates detailed truck routing data for Belgium logistics using GraphHopper, with support for diesel, electric, and hydrogen truck profiles. The data includes elevation profiles, slope analysis, and road characteristics for building flow networks.

## Features

- **306 OD pairs** across 18 Belgium locations
- **3 truck profiles**: Diesel, EV, H2 with realistic slope penalties
- **Node extraction**: Waypoints every 10 miles for graph creation
- **Route visualization**: Interactive maps with terrain and connectivity views
- **Comprehensive attributes**: Elevation, slopes, road classes, surfaces, environments

## Generated Data

- `Data/route_paths.json` - 306 routes with node coordinates (206 KB)
- `Data/master_locations.csv` - 18 Belgium logistics locations
- `Data/truck_fleet_config.yaml` - Master truck configuration

## Usage

### Generate routes
```bash
python Code/batch_generate_routes.py --alternatives 3 --profile truck_diesel
```

### Extract paths for graph
```bash
python Code/extract_route_paths.py --alternative 1
```

### Visualize network
```bash
python Code/visualize_all_routes.py --color-mode slope
```

## Requirements

- Python 3.13+
- GraphHopper 9.1
- See requirements.txt

## Next Steps

- Traffic condition overlay
- Graph structure generation
- Multi-profile flow network
- FCGL training integration
