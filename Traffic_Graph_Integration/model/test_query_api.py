"""
Test Script for FCGL Query API
Demonstrates various query capabilities
"""

from fcgl_query_api import FCGLQueryAPI

print("="*80)
print("FCGL QUERY API - TEST SUITE")
print("="*80)

# Initialize API
api = FCGLQueryAPI()

print("\n" + "="*80)
print("TEST 1: Greedy Deterministic Path")
print("="*80)
result = api.get_greedy_path("S0", "D2")
print(f"Path: {result['path_str']}")
print(f"Cost: ${result['cost']:.2f}, CO2: {result['co2']:.2f} kg")
print(f"Steps: {result['steps']}, Terminal: {result['terminal']}")

print("\n" + "="*80)
print("TEST 2: Min-Cost Optimal Baseline")
print("="*80)
result = api.get_min_cost_path("S1", "D4")
print(f"Path: {result['path_str']}")
print(f"Cost: ${result['cost']:.2f}, CO2: {result['co2']:.2f} kg")
print(f"Distance: {result['distance']:.2f} km, Time: {result['time']:.2f} hours")

print("\n" + "="*80)
print("TEST 3: Terminal Probability Distribution")
print("="*80)
result = api.get_all_terminal_probabilities("S0", num_samples=200)
print(f"Source: {result['source']}, Samples: {result['total_samples']}")
print(f"Most likely: {result['most_likely']}")
for sink, data in sorted(result['probabilities'].items(), key=lambda x: x[1]['percentage'], reverse=True):
    print(f"  {sink}: {data['percentage']:.2f}% ({data['count']} visits)")

print("\n" + "="*80)
print("TEST 4: Expected Cost Statistics")
print("="*80)
result = api.get_expected_cost("S1", "D4", num_samples=100)
print(f"Source: {result['source']} → Sink: {result['sink']}")
print(f"Samples reaching target: {result['count']}/{result.get('total_samples', 100)} ({result.get('sample_rate', 0)}%)")
print(f"Expected Cost: ${result['cost']['mean']:.2f} ± ${result['cost']['std']:.2f}")
print(f"  Min: ${result['cost']['min']:.2f}, Max: ${result['cost']['max']:.2f}")

print("\n" + "="*80)
print("TEST 5: Most Probable Path")
print("="*80)
result = api.get_most_probable_path("S0", "D2", num_samples=150)
print(f"Most common path: {result['path_str']}")
print(f"Frequency: {result['frequency']}/{result['total_paths']} ({result['percentage']:.2f}%)")
print(f"Cost: ${result['cost']:.2f}")

print("\n" + "="*80)
print("TEST 6: Natural Language Query")
print("="*80)
result = api.ask("probability of reaching D3 from S1")
print(f"Query parsed: {result.get('query_type', 'unknown')}")
print(f"Probability: {result['probability']:.4f} ({result['percentage']:.2f}%)")
print(f"Visits: {result['count']}/{result['total_samples']}")

print("\n" + "="*80)
print("TEST 7: Compare Greedy vs Optimal")
print("="*80)
greedy = api.get_greedy_path("S1", "D4")
optimal = api.get_min_cost_path("S1", "D4")
print(f"Greedy Path:")
print(f"  Cost: ${greedy['cost']:.2f}, Steps: {greedy['steps']}")
print(f"Optimal Path (Dijkstra):")
print(f"  Cost: ${optimal['cost']:.2f}, Steps: {optimal['steps']}")
cost_diff = greedy['cost'] - optimal['cost']
pct_diff = (cost_diff / optimal['cost'] * 100) if optimal['cost'] > 0 else 0
print(f"Difference: ${cost_diff:.2f} ({pct_diff:+.2f}%)")

print("\n" + "="*80)
print("✓ All Tests Complete!")
print("="*80)
