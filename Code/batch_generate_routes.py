#!/usr/bin/env python3
"""
Batch generate routes for all OD pairs from master_locations.csv.
Generates 18×18 location pairs (excluding self-routes) with multiple alternatives.
"""

import pandas as pd
import sys
import os
import time
from datetime import datetime
from generate_route import generate_routes
import json

def load_locations(csv_path: str = "../Data/master_locations.csv") -> pd.DataFrame:
    """Load location database."""
    return pd.read_csv(csv_path)

def batch_generate_all_routes(
    profile: str = "truck_diesel",
    num_alternatives: int = 3,
    output_dir: str = "../Maps/batch_routes",
    skip_existing: bool = True
):
    """
    Generate routes for all location pairs.
    
    Args:
        profile: GraphHopper profile (truck_diesel, truck_ev, truck_h2)
        num_alternatives: Number of alternative routes per OD pair
        output_dir: Directory to save outputs
        skip_existing: Skip if route JSON already exists
    """
    print("\n" + "="*80)
    print(f"BATCH ROUTE GENERATION - Belgium Logistics Network")
    print(f"Profile: {profile} | Alternatives: {num_alternatives}")
    print("="*80 + "\n")
    
    # Load locations
    locations_df = load_locations()
    locations = locations_df['Location'].tolist()
    num_locations = len(locations)
    
    print(f"Loaded {num_locations} locations:")
    for i, loc in enumerate(locations, 1):
        print(f"  {i:2d}. {loc}")
    
    # Calculate total OD pairs (exclude self-routes)
    total_pairs = num_locations * (num_locations - 1)
    print(f"\nTotal OD pairs: {total_pairs} ({num_locations}×{num_locations-1})")
    print(f"Expected routes: {total_pairs * num_alternatives}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track progress
    successful = 0
    failed = 0
    skipped = 0
    failed_pairs = []
    
    start_time = time.time()
    
    # Generate routes for all pairs
    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            # Skip self-routes
            if origin == destination:
                continue
            
            pair_num = i * num_locations + j
            progress = (successful + failed + skipped) / total_pairs * 100
            
            print(f"\n[{pair_num}/{total_pairs}] ({progress:.1f}%) {origin} → {destination}")
            
            # Check if already exists
            safe_origin = origin.replace(' ', '_').replace(',', '')
            safe_dest = destination.replace(' ', '_').replace(',', '')
            pattern = f"{safe_origin}_to_{safe_dest}_*.json"
            
            existing_files = [f for f in os.listdir(output_dir) if f.startswith(f"{safe_origin}_to_{safe_dest}_")]
            
            if skip_existing and existing_files:
                print(f"  ⏭️  Skipping (already exists): {existing_files[0]}")
                skipped += 1
                continue
            
            try:
                # Generate route
                output_data = generate_routes(
                    source=origin,
                    destination=destination,
                    profile=profile,
                    num_alternatives=num_alternatives,
                    output_dir=output_dir
                )
                successful += 1
                
                # Small delay to avoid overwhelming GraphHopper
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                failed += 1
                failed_pairs.append((origin, destination, str(e)))
                continue
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    
    print("\n" + "="*80)
    print("BATCH GENERATION COMPLETE")
    print("="*80)
    print(f"Total pairs:     {total_pairs}")
    print(f"Successful:      {successful} ✓")
    print(f"Skipped:         {skipped} ⏭️")
    print(f"Failed:          {failed} ❌")
    print(f"Time elapsed:    {elapsed_min:.1f} minutes")
    print(f"Avg per route:   {elapsed_time/max(successful, 1):.1f} seconds")
    print(f"Output dir:      {output_dir}")
    print("="*80 + "\n")
    
    # Save failed pairs log
    if failed_pairs:
        log_file = os.path.join(output_dir, f"failed_pairs_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w') as f:
            json.dump({
                "profile": profile,
                "num_alternatives": num_alternatives,
                "total_failed": failed,
                "failed_pairs": [
                    {"origin": o, "destination": d, "error": e} 
                    for o, d, e in failed_pairs
                ]
            }, f, indent=2)
        print(f"Failed pairs log saved to: {log_file}\n")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"batch_summary_{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "profile": profile,
            "num_alternatives": num_alternatives,
            "total_pairs": total_pairs,
            "successful": successful,
            "skipped": skipped,
            "failed": failed,
            "elapsed_minutes": round(elapsed_min, 2),
            "avg_seconds_per_route": round(elapsed_time/max(successful, 1), 2),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch generate routes for all OD pairs")
    parser.add_argument("--profile", default="truck_diesel", 
                       choices=["truck_diesel", "truck_ev", "truck_h2"],
                       help="GraphHopper profile to use")
    parser.add_argument("--alternatives", type=int, default=3,
                       help="Number of alternative routes per OD pair (1-5)")
    parser.add_argument("--output-dir", default="../Maps/batch_routes",
                       help="Output directory for route files")
    parser.add_argument("--no-skip", action="store_true",
                       help="Regenerate even if route already exists")
    
    args = parser.parse_args()
    
    try:
        batch_generate_all_routes(
            profile=args.profile,
            num_alternatives=args.alternatives,
            output_dir=args.output_dir,
            skip_existing=not args.no_skip
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        sys.exit(1)
