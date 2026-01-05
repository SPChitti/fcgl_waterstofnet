"""
Query trained FCGL model with detailed metrics
Usage: python3 query_with_details.py [source] [destination]
"""

import sys
from fcgl_query_api import FCGLQueryAPI

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 query_with_details.py [source] [destination]")
        print("Example: python3 query_with_details.py S0 D2")
        print("         python3 query_with_details.py S1 D4")
        sys.exit(1)
    
    source = sys.argv[1].upper()
    destination = sys.argv[2].upper()
    
    print("="*80)
    print(f"FCGL ROUTE ANALYSIS: {source} → {destination}")
    print("="*80)
    
    # Initialize API
    api = FCGLQueryAPI()
    
    print("\n" + "="*80)
    print("1️⃣  TRAINED POLICY (Greedy Path)")
    print("="*80)
    
    greedy = api.get_greedy_path_with_details(source, destination)
    
    if 'error' in greedy:
        print(f"❌ Error: {greedy['error']}")
    else:
        print(f"\n📍 Route: {greedy['path'][0]} → {greedy['terminal']}")
        if len(greedy['path']) <= 10:
            print(f"🛣️  Path: {' → '.join(greedy['path'])}")
        else:
            print(f"🛣️  Path: {' → '.join(greedy['path'][:5])} ... {' → '.join(greedy['path'][-3:])}")
        print(f"📊 Steps: {greedy['steps']}")
        
        api.metrics_calculator.print_detailed_metrics(greedy['detailed_metrics'])
    
    print("\n" + "="*80)
    print("2️⃣  OPTIMAL BASELINE (Dijkstra's Algorithm)")
    print("="*80)
    
    optimal = api.get_min_cost_path_with_details(source, destination)
    
    if 'error' in optimal:
        print(f"❌ Error: {optimal['error']}")
    else:
        print(f"\n📍 Route: {optimal['source']} → {optimal['sink']}")
        if len(optimal['path']) <= 10:
            print(f"🛣️  Path: {' → '.join(optimal['path'])}")
        else:
            print(f"🛣️  Path: {' → '.join(optimal['path'][:5])} ... {' → '.join(optimal['path'][-3:])}")
        print(f"📊 Steps: {optimal['steps']}")
        
        api.metrics_calculator.print_detailed_metrics(optimal['detailed_metrics'])
    
    # Comparison
    if 'error' not in greedy and 'error' not in optimal:
        print("\n" + "="*80)
        print("📊 COMPARISON: Trained Policy vs Optimal Baseline")
        print("="*80)
        
        g_cost = greedy['detailed_metrics']['cost_metrics']['total_route_cost']
        o_cost = optimal['detailed_metrics']['cost_metrics']['total_route_cost']
        diff = g_cost - o_cost
        pct_diff = (diff / o_cost * 100) if o_cost > 0 else 0
        
        g_h2 = greedy['detailed_metrics']['consumption_metrics']['h2_consumed']
        o_h2 = optimal['detailed_metrics']['consumption_metrics']['h2_consumed']
        
        g_time = greedy['detailed_metrics']['operational_metrics']['total_operational_time']
        o_time = optimal['detailed_metrics']['operational_metrics']['total_operational_time']
        
        print(f"\n{'Metric':<30} {'Policy':>15} {'Optimal':>15} {'Difference':>20}")
        print("─"*85)
        print(f"{'Total Cost (€)':<30} {g_cost:>15.2f} {o_cost:>15.2f} {diff:>15.2f} ({pct_diff:+.1f}%)")
        print(f"{'H₂ Consumed (kg)':<30} {g_h2:>15.2f} {o_h2:>15.2f} {g_h2-o_h2:>15.2f}")
        print(f"{'Time (hours)':<30} {g_time:>15.2f} {o_time:>15.2f} {g_time-o_time:>15.2f}")
        print(f"{'Steps':<30} {greedy['steps']:>15} {optimal['steps']:>15} {greedy['steps']-optimal['steps']:>15}")
        
        print("\n" + "="*80)
        if diff < -1.0:
            savings = -diff
            pct_savings = -pct_diff
            print(f"✅ POLICY WINS: €{savings:.2f} cheaper ({pct_savings:.1f}% savings)")
        elif diff > 1.0:
            print(f"⚠️  OPTIMAL WINS: €{diff:.2f} cheaper ({abs(pct_diff):.1f}% savings)")
        else:
            print("🤝 EQUIVALENT: Costs are nearly identical")
        print("="*80)

if __name__ == "__main__":
    main()
