"""
Test Detailed Metrics Queries
Demonstrates comprehensive route analysis similar to the image
"""

from fcgl_query_api import FCGLQueryAPI

print("="*80)
print("FCGL DETAILED METRICS DEMONSTRATION")
print("="*80)

# Initialize API
api = FCGLQueryAPI()

print("\n" + "="*80)
print("QUERY 1: Greedy Path from S0 to D2 (with detailed metrics)")
print("="*80)

result = api.get_greedy_path_with_details("S0", "D2")

print(f"\n📍 Route: {result['path'][0]} → {result['terminal']}")
print(f"🛣️  Path: {' → '.join(result['path'][:5])} ... {' → '.join(result['path'][-3:])}")
print(f"📊 Steps: {result['steps']}")

# Print detailed metrics
api.metrics_calculator.print_detailed_metrics(result['detailed_metrics'])

print("\n" + "="*80)
print("QUERY 2: Optimal Path from S1 to D4 (with detailed metrics)")
print("="*80)

result = api.get_min_cost_path_with_details("S1", "D4")

print(f"\n📍 Route: {result['source']} → {result['sink']}")
print(f"🛣️  Path: {' → '.join(result['path'][:5])} ... {' → '.join(result['path'][-3:])}")
print(f"📊 Steps: {result['steps']}")

# Print detailed metrics
api.metrics_calculator.print_detailed_metrics(result['detailed_metrics'])

print("\n" + "="*80)
print("QUERY 3: Compare Greedy vs Optimal (Detailed Cost Breakdown)")
print("="*80)

greedy = api.get_greedy_path_with_details("S1", "D4")
optimal = api.get_min_cost_path_with_details("S1", "D4")

print("\n📊 COST COMPARISON:")
print(f"\n{'Metric':<30} {'Greedy':>15} {'Optimal':>15} {'Diff':>15}")
print("─"*80)

g_cost = greedy['detailed_metrics']['cost_metrics']
o_cost = optimal['detailed_metrics']['cost_metrics']

metrics = [
    ('Fuel/Energy Cost', 'fuel_energy_cost'),
    ('Driver Cost', 'driver_cost'),
    ('Vehicle Depreciation', 'vehicle_depreciation'),
    ('Maintenance', 'maintenance'),
    ('TOTAL ROUTE COST', 'total_route_cost')
]

for label, key in metrics:
    g_val = g_cost[key]
    o_val = o_cost[key]
    diff = g_val - o_val
    diff_pct = (diff / o_val * 100) if o_val > 0 else 0
    
    if label == 'TOTAL ROUTE COST':
        print("─"*80)
    
    print(f"{label:<30} €{g_val:>13.2f} €{o_val:>13.2f} €{diff:>12.2f} ({diff_pct:+.1f}%)")

print("\n📊 CONSUMPTION COMPARISON:")
print(f"\n{'Metric':<30} {'Greedy':>15} {'Optimal':>15} {'Diff':>15}")
print("─"*80)

g_cons = greedy['detailed_metrics']['consumption_metrics']
o_cons = optimal['detailed_metrics']['consumption_metrics']

print(f"{'H₂ Consumed (kg)':<30} {g_cons['h2_consumed']:>15.2f} {o_cons['h2_consumed']:>15.2f} {g_cons['h2_consumed']-o_cons['h2_consumed']:>15.2f}")
print(f"{'Avg per km (kg/km)':<30} {g_cons['average_per_km']:>15.4f} {o_cons['average_per_km']:>15.4f} {g_cons['average_per_km']-o_cons['average_per_km']:>15.4f}")
print(f"{'Fuel Efficiency (%)':<30} {g_cons['fuel_efficiency']:>15.1f} {o_cons['fuel_efficiency']:>15.1f} {g_cons['fuel_efficiency']-o_cons['fuel_efficiency']:>15.1f}")
print(f"{'Refueling Stops':<30} {g_cons['refueling_stops']:>15} {o_cons['refueling_stops']:>15} {g_cons['refueling_stops']-o_cons['refueling_stops']:>15}")

print("\n📊 OPERATIONAL COMPARISON:")
print(f"\n{'Metric':<30} {'Greedy':>15} {'Optimal':>15} {'Diff':>15}")
print("─"*80)

g_ops = greedy['detailed_metrics']['operational_metrics']
o_ops = optimal['detailed_metrics']['operational_metrics']

print(f"{'Driving Time (hrs)':<30} {g_ops['driving_time']:>15.2f} {o_ops['driving_time']:>15.2f} {g_ops['driving_time']-o_ops['driving_time']:>15.2f}")
print(f"{'Refueling Time (hrs)':<30} {g_ops['refueling_time']:>15.2f} {o_ops['refueling_time']:>15.2f} {g_ops['refueling_time']-o_ops['refueling_time']:>15.2f}")
print(f"{'Driver Breaks':<30} {g_ops['driver_breaks']:>15} {o_ops['driver_breaks']:>15} {g_ops['driver_breaks']-o_ops['driver_breaks']:>15}")
print(f"{'Total Operational (hrs)':<30} {g_ops['total_operational_time']:>15.2f} {o_ops['total_operational_time']:>15.2f} {g_ops['total_operational_time']-o_ops['total_operational_time']:>15.2f}")

# Summary
total_diff = g_cost['total_route_cost'] - o_cost['total_route_cost']
pct_diff = (total_diff / o_cost['total_route_cost'] * 100) if o_cost['total_route_cost'] > 0 else 0

print("\n" + "="*80)
print("💡 SUMMARY")
print("="*80)
print(f"Cost Savings: €{-total_diff:.2f} ({-pct_diff:.1f}%)")
print(f"H₂ Saved: {o_cons['h2_consumed']-g_cons['h2_consumed']:.2f} kg")
print(f"Time Saved: {(o_ops['total_operational_time']-g_ops['total_operational_time'])*60:.1f} minutes")

if total_diff < 0:
    print(f"\n✅ Greedy path is BETTER by €{-total_diff:.2f}!")
else:
    print(f"\n⚠️ Optimal path is better by €{total_diff:.2f}")

print("="*80)
