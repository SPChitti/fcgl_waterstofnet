#!/usr/bin/env python3
"""
Generate GraphHopper custom model JSON files from truck fleet config.
Reads truck_fleet_config.yaml and creates truck_diesel.json, truck_ev.json, truck_h2.json
"""

import yaml
import json
from pathlib import Path

def load_config(config_path: str = "truck_fleet_config.yaml"):
    """Load truck fleet configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_graphhopper_custom_model(truck_config: dict) -> dict:
    """Convert truck config to GraphHopper custom model format."""
    
    physical = truck_config['physical']
    routing = truck_config['routing']
    
    # Build priority rules
    priority_rules = []
    for rule in routing['priority']:
        if 'default' in rule and rule['default']:
            priority_rules.append({
                "else": "",
                "multiply_by": rule['multiply_by']
            })
        else:
            rule_type = "if" if not priority_rules else "else_if"
            priority_rules.append({
                rule_type: rule['condition'],
                "multiply_by": rule['multiply_by']
            })
    
    # Build speed limit rules
    speed_rules = []
    for rule in routing['speed_limits']:
        if 'default' in rule and rule['default']:
            speed_rules.append({
                "else": "",
                "limit_to": rule['limit_to']
            })
        else:
            rule_type = "if" if not speed_rules else "else_if"
            speed_rules.append({
                rule_type: rule['condition'],
                "limit_to": rule['limit_to']
            })
    
    # Build custom model (GraphHopper only accepts standard fields)
    custom_model = {
        "distance_influence": routing['distance_influence'],
        "priority": priority_rules,
        "speed": speed_rules
    }
    
    return custom_model

def main():
    """Main entry point."""
    print("=" * 80)
    print("GraphHopper Custom Model Generator")
    print("=" * 80)
    
    # Load config
    print("\n1. Loading truck fleet config...")
    config = load_config()
    print(f"   ✓ Found {len(config['truck_types'])} truck types")
    
    # Generate custom models
    print("\n2. Generating GraphHopper custom model files...")
    
    for truck_key, truck_config in config['truck_types'].items():
        truck_name = truck_config['name']
        output_file = f"{truck_name}.json"
        
        custom_model = generate_graphhopper_custom_model(truck_config)
        
        with open(output_file, 'w') as f:
            json.dump(custom_model, f, indent=2)
        
        print(f"   ✓ Generated {output_file}")
        print(f"     - Range: {truck_config['physical']['range_km']}km")
        print(f"     - Refuel: {truck_config['physical'].get('refuel_time_minutes') or truck_config['physical'].get('recharge_time_minutes')}min")
    
    print("\n" + "=" * 80)
    print("✓ Custom model files generated successfully")
    print("=" * 80)
    print("\nNext step: Rebuild GraphHopper graph with:")
    print("  ./rebuild_graphhopper.sh")
    print()

if __name__ == "__main__":
    main()
