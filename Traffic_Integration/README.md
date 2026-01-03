# Traffic Integration Project

This directory contains the workflow for integrating historical traffic data with truck routing.

## Folder Structure

```
Traffic_Integration/
├── selected_routes/        # 3 representative routes (JSON files)
├── midway_points/          # Extracted midway points for each route
├── traffic_data/           # TomTom API responses (historical traffic)
├── config/                 # Traffic configuration files for GraphHopper
├── scripts/                # Python scripts for the workflow
└── outputs/                # Final traffic-aware routes
```

## Workflow

### Phase 1: Route Selection & Midway Point Extraction
1. Select 3 representative routes from existing 306 routes
2. Extract 4-6 midway points per route
3. Store in `midway_points/`

### Phase 2: Traffic Data Collection
1. Query TomTom Historical Traffic API for each segment
2. Get 24h × 7days traffic patterns
3. Store raw responses in `traffic_data/`

### Phase 3: Configuration Creation
1. Map TomTom segments to GraphHopper edges
2. Create time-based speed multiplier configs
3. Store in `config/`

### Phase 4: Route Generation
1. Modify GraphHopper weighting with traffic
2. Generate new routes for different time slots
3. Store results in `outputs/`

## API Budget
- 3 routes × ~5 midpoints × 24h × 7days = ~2,520 queries
- Within TomTom free tier (2,500/day over 2 days)
