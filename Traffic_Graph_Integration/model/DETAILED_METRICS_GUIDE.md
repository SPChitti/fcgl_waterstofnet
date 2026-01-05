# FCGL Detailed Metrics Query System

## Overview
Enhanced query system that provides comprehensive route analysis similar to industry-standard routing systems.

## Available Metrics

### 💰 Cost Metrics
- **Fuel/Energy Cost**: Hydrogen consumption × €13.30/kg
- **Driver Cost**: Operational time × €30.86/hour (EU average)
- **Vehicle Depreciation**: Distance-based (€0.12-0.18/km by truck type)
- **Maintenance**: Distance-based (€0.08-0.12/km by truck type)
- **Total Route Cost**: Sum of all cost components

### ⚡ Consumption Metrics
- **H₂ Consumed**: Total hydrogen used (kg)
- **Average per km**: H₂ consumption rate (kg/km)
- **Fuel Efficiency**: Actual vs theoretical consumption (%)
- **Refueling Stops**: Number of stops needed based on tank capacity

### ⏱️ Operational Metrics
- **Driving Time**: Pure travel time (hours)
- **Stop Time**: Additional stops for refueling/breaks
- **Refueling Time**: Time spent refueling (12 min per stop)
- **Idle Time**: Additional stop time overhead
- **Driver Breaks**: Mandatory breaks per EU regulations (45 min every 4.5 hours)
- **Total Operational Time**: Complete end-to-end time

### 📊 Route Summary
- **Total Distance**: Route length (km)
- **Primary Truck Type**: small/medium/heavy
- **Number of Edges**: Route complexity

## Usage

### Command Line Interface
```bash
# Query any route with detailed breakdown
python3 query_with_details.py S0 D2
python3 query_with_details.py S1 D4
python3 query_with_details.py S1 D3
```

**Output**: Shows both trained policy path and optimal baseline, with full comparison.

### Python API
```python
from fcgl_query_api import FCGLQueryAPI

# Initialize
api = FCGLQueryAPI()

# Get detailed metrics for greedy path
result = api.get_greedy_path_with_details("S0", "D2")
api.metrics_calculator.print_detailed_metrics(result['detailed_metrics'])

# Get detailed metrics for optimal path
result = api.get_min_cost_path_with_details("S1", "D4")
api.metrics_calculator.print_detailed_metrics(result['detailed_metrics'])

# Get metrics for any path
path = ["S0", "5", "6", "23", "D2"]
metrics = api.get_detailed_metrics_for_path(path)
```

## Cost Calculation Details

### Fuel/Energy Cost
```
H₂ consumption (kg/km) = base_rate × (1 + slope_factor) × (1 + congestion_factor)
- Base rates: small=0.050, medium=0.065, heavy=0.080 kg/km
- Slope adjustment: +2% per 1% gradient
- Congestion adjustment: +0.1% per 1% congestion
- Cost = total_H₂ × €13.30/kg
```

### Driver Cost
```
Total time = driving + refueling + breaks
- Driving: from edge travel times
- Refueling: 0.2 hours per stop (12 minutes)
- Breaks: 0.75 hours every 4.5 hours of driving
- Cost = total_time × €30.86/hour
```

### Vehicle Costs
```
Depreciation: distance × rate (€0.12-0.18/km)
Maintenance: distance × rate (€0.08-0.12/km)
Rates vary by truck type (small < medium < heavy)
```

### Refueling Logic
```
Tank capacities: small=10kg, medium=15kg, heavy=20kg
Stop triggered when: cumulative_H₂ + next_edge_H₂ > tank_capacity
```

## Files

- **detailed_metrics.py**: Core metrics calculator class
- **query_with_details.py**: Command-line interface
- **test_detailed_metrics.py**: Comprehensive test suite
- **fcgl_query_api.py**: Enhanced API with detailed metrics support

## Example Output

```
💰 COST METRICS
  Fuel / Energy Cost:      €   89.00
  Driver Cost:             €   82.60
  Vehicle Depreciation:    €   16.06
  Maintenance:             €   10.71
  ────────────────────────────────────────
  Total Route Cost:        €  198.37

⚡ CONSUMPTION METRICS
  H₂ Consumed:                 6.69 kg
  Average per km:            0.0500 kg/km
  Fuel Efficiency:            100.0%
  Refueling Stops:                0 stops

⏱️  OPERATIONAL METRICS
  Driving Time:                2.68 hrs
  Stop Time:                   0.00 hrs
  Refueling Time:              0.00 hrs
  Idle Time:                   0.00 hrs
  Driver Breaks:                  0 breaks
  ────────────────────────────────────────
  Total Operational Time:      2.68 hrs
```

## Matching Industry Standards

These metrics match industry-standard routing systems and provide:
- ✓ Detailed cost breakdown (fuel, driver, vehicle)
- ✓ Hydrogen consumption tracking
- ✓ Operational time breakdown
- ✓ Refueling stop calculations
- ✓ Driver break compliance (EU regulations)
- ✓ Fuel efficiency metrics

Ready for production deployment and integration with fleet management systems!
