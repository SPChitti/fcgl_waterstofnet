"""
Detailed Route Metrics Calculator
Computes comprehensive cost, consumption, and operational metrics
"""

from typing import Dict, List, Any, Tuple
import math


class DetailedMetricsCalculator:
    """Calculate detailed route metrics from path and edges"""
    
    # Cost parameters (€ per unit)
    HYDROGEN_COST_PER_KG = 13.30  # €/kg H₂
    DRIVER_COST_PER_HOUR = 30.86  # €/hour (average EU driver wage)
    
    # Vehicle depreciation (€/km by truck type)
    DEPRECIATION = {
        'small': 0.12,   # €/km
        'medium': 0.15,  # €/km
        'heavy': 0.18    # €/km
    }
    
    # Maintenance cost (€/km by truck type)
    MAINTENANCE = {
        'small': 0.08,   # €/km
        'medium': 0.10,  # €/km
        'heavy': 0.12    # €/km
    }
    
    # Hydrogen consumption (kg/km by truck type - base rates)
    H2_CONSUMPTION = {
        'small': 0.050,   # kg/km
        'medium': 0.065,  # kg/km
        'heavy': 0.080    # kg/km
    }
    
    # Refueling parameters
    TANK_CAPACITY = {
        'small': 10.0,    # kg H₂
        'medium': 15.0,   # kg H₂
        'heavy': 20.0     # kg H₂
    }
    REFUEL_TIME_HOURS = 0.2  # 12 minutes per refueling
    REFUEL_STOP_TIME = 0.1   # Additional stop time (6 minutes)
    
    # Driver break rules (EU regulations)
    MAX_DRIVING_TIME = 4.5    # hours before mandatory break
    BREAK_DURATION = 0.75     # 45 minutes
    
    
    def __init__(self):
        pass
    
    def calculate_detailed_metrics(self, path_edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a route
        
        Args:
            path_edges: List of edge dictionaries from query result
            
        Returns:
            Dictionary with detailed cost, consumption, and operational metrics
        """
        if not path_edges:
            return self._empty_metrics()
        
        # Initialize accumulators
        total_distance = 0.0
        total_time = 0.0
        total_h2_consumed = 0.0
        
        fuel_cost = 0.0
        driver_cost = 0.0
        depreciation_cost = 0.0
        maintenance_cost = 0.0
        
        truck_types_used = []
        cumulative_h2 = 0.0
        refueling_stops = 0
        refueling_time = 0.0
        idle_time = 0.0
        
        # Get truck type from first edge (assume consistent or track changes)
        primary_truck_type = path_edges[0].get('mode_type', 
                                               path_edges[0].get('truck_type', 'medium'))
        tank_capacity = self.TANK_CAPACITY.get(primary_truck_type, 15.0)
        
        # Process each edge
        for edge in path_edges:
            distance = edge.get('dist', edge.get('distance_km', 0.0))
            time = edge.get('time', edge.get('travel_time_hours', 0.0))
            truck_type = edge.get('mode_type', edge.get('truck_type', primary_truck_type))
            
            # Base H2 consumption
            h2_rate = self.H2_CONSUMPTION.get(truck_type, 0.065)
            
            # Adjust for slopes
            avg_slope = edge.get('avg_slope_pct', 0.0)
            if avg_slope > 0:
                # Increase consumption by 2% per 1% slope
                h2_rate *= (1 + 0.02 * avg_slope)
            
            # Adjust for congestion
            congestion = edge.get('morning_congestion_pct', 
                                 edge.get('offpeak_congestion_pct', 0.0))
            if congestion > 0:
                # Increase consumption by 1% per 10% congestion
                h2_rate *= (1 + 0.001 * congestion)
            
            h2_consumed = h2_rate * distance
            
            # Check if refueling needed
            if cumulative_h2 + h2_consumed > tank_capacity:
                refueling_stops += 1
                cumulative_h2 = h2_consumed  # Reset after refuel
                refueling_time += self.REFUEL_TIME_HOURS
                idle_time += self.REFUEL_STOP_TIME
            else:
                cumulative_h2 += h2_consumed
            
            # Accumulate
            total_distance += distance
            total_time += time
            total_h2_consumed += h2_consumed
            truck_types_used.append(truck_type)
            
            # Cost components
            fuel_cost += h2_consumed * self.HYDROGEN_COST_PER_KG
            depreciation_cost += distance * self.DEPRECIATION.get(truck_type, 0.15)
            maintenance_cost += distance * self.MAINTENANCE.get(truck_type, 0.10)
        
        # Driver cost (includes driving + refueling + breaks)
        driver_breaks = math.floor(total_time / self.MAX_DRIVING_TIME)
        break_time = driver_breaks * self.BREAK_DURATION
        total_operational_time = total_time + refueling_time + break_time
        driver_cost = total_operational_time * self.DRIVER_COST_PER_HOUR
        
        # Total costs
        total_cost = fuel_cost + driver_cost + depreciation_cost + maintenance_cost
        
        # Fuel efficiency (actual vs theoretical)
        theoretical_h2 = total_distance * self.H2_CONSUMPTION.get(primary_truck_type, 0.065)
        fuel_efficiency = (theoretical_h2 / total_h2_consumed * 100) if total_h2_consumed > 0 else 100.0
        
        return {
            'cost_metrics': {
                'fuel_energy_cost': round(fuel_cost, 2),
                'driver_cost': round(driver_cost, 2),
                'vehicle_depreciation': round(depreciation_cost, 2),
                'maintenance': round(maintenance_cost, 2),
                'total_route_cost': round(total_cost, 2)
            },
            'consumption_metrics': {
                'h2_consumed': round(total_h2_consumed, 2),
                'average_per_km': round(total_h2_consumed / total_distance, 4) if total_distance > 0 else 0.0,
                'fuel_efficiency': round(fuel_efficiency, 1),
                'refueling_stops': refueling_stops
            },
            'operational_metrics': {
                'driving_time': round(total_time, 2),
                'stop_time': round(idle_time, 2),
                'refueling_time': round(refueling_time, 2),
                'idle_time': round(idle_time, 2),
                'driver_breaks': driver_breaks,
                'total_operational_time': round(total_operational_time, 2)
            },
            'summary': {
                'total_distance': round(total_distance, 2),
                'primary_truck_type': primary_truck_type,
                'num_edges': len(path_edges)
            }
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'cost_metrics': {
                'fuel_energy_cost': 0.0,
                'driver_cost': 0.0,
                'vehicle_depreciation': 0.0,
                'maintenance': 0.0,
                'total_route_cost': 0.0
            },
            'consumption_metrics': {
                'h2_consumed': 0.0,
                'average_per_km': 0.0,
                'fuel_efficiency': 100.0,
                'refueling_stops': 0
            },
            'operational_metrics': {
                'driving_time': 0.0,
                'stop_time': 0.0,
                'refueling_time': 0.0,
                'idle_time': 0.0,
                'driver_breaks': 0,
                'total_operational_time': 0.0
            },
            'summary': {
                'total_distance': 0.0,
                'primary_truck_type': 'unknown',
                'num_edges': 0
            }
        }
    
    def print_detailed_metrics(self, metrics: Dict[str, Any]) -> None:
        """Pretty print detailed metrics"""
        print("\n" + "="*60)
        print("💰 COST METRICS")
        print("="*60)
        cm = metrics['cost_metrics']
        print(f"  Fuel / Energy Cost:      €{cm['fuel_energy_cost']:>8.2f}")
        print(f"  Driver Cost:             €{cm['driver_cost']:>8.2f}")
        print(f"  Vehicle Depreciation:    €{cm['vehicle_depreciation']:>8.2f}")
        print(f"  Maintenance:             €{cm['maintenance']:>8.2f}")
        print(f"  {'─'*40}")
        print(f"  Total Route Cost:        €{cm['total_route_cost']:>8.2f}")
        
        print("\n" + "="*60)
        print("⚡ CONSUMPTION METRICS")
        print("="*60)
        cons = metrics['consumption_metrics']
        print(f"  H₂ Consumed:             {cons['h2_consumed']:>8.2f} kg")
        print(f"  Average per km:          {cons['average_per_km']:>8.4f} kg/km")
        print(f"  Fuel Efficiency:         {cons['fuel_efficiency']:>8.1f}%")
        print(f"  Refueling Stops:         {cons['refueling_stops']:>8} stop{'s' if cons['refueling_stops'] != 1 else ''}")
        
        print("\n" + "="*60)
        print("⏱️  OPERATIONAL METRICS")
        print("="*60)
        ops = metrics['operational_metrics']
        print(f"  Driving Time:            {ops['driving_time']:>8.2f} hrs")
        print(f"  Stop Time:               {ops['stop_time']:>8.2f} hrs")
        print(f"  Refueling Time:          {ops['refueling_time']:>8.2f} hrs")
        print(f"  Idle Time:               {ops['idle_time']:>8.2f} hrs")
        print(f"  Driver Breaks:           {ops['driver_breaks']:>8} break{'s' if ops['driver_breaks'] != 1 else ''}")
        print(f"  {'─'*40}")
        print(f"  Total Operational Time:  {ops['total_operational_time']:>8.2f} hrs")
        
        print("\n" + "="*60)
        print("📊 ROUTE SUMMARY")
        print("="*60)
        summ = metrics['summary']
        print(f"  Total Distance:          {summ['total_distance']:>8.2f} km")
        print(f"  Primary Truck Type:      {summ['primary_truck_type']:>8}")
        print(f"  Number of Edges:         {summ['num_edges']:>8}")
        print("="*60)
