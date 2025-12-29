#!/bin/bash
# Rebuild GraphHopper with updated truck configurations
# Run this script after modifying truck_fleet_config.yaml

set -e

cd "$(dirname "$0")"

echo "================================================================================"
echo "GraphHopper Rebuild Script - Belgium Logistics"
echo "================================================================================"

# 1. Generate custom model files from config
echo ""
echo "Step 1: Generating GraphHopper custom model files from truck_fleet_config.yaml"
echo "--------------------------------------------------------------------------------"
python3 generate_custom_models.py

# 2. Stop GraphHopper service
echo ""
echo "Step 2: Stopping GraphHopper service..."
echo "--------------------------------------------------------------------------------"
sudo systemctl stop graphhopper
sleep 2
echo "✓ Service stopped"

# 3. Remove old graph cache
echo ""
echo "Step 3: Removing old graph cache..."
echo "--------------------------------------------------------------------------------"
if [ -d "graph-cache" ]; then
    rm -rf graph-cache
    echo "✓ Old cache removed"
else
    echo "✓ No cache to remove"
fi

# 4. Rebuild graph
echo ""
echo "Step 4: Building new graph with updated profiles..."
echo "--------------------------------------------------------------------------------"
echo "This will take 5-10 minutes..."
echo ""

java -Xmx4g -Xms1g -jar graphhopper-web-9.1.jar import config.yml

# 5. Restart service
echo ""
echo "Step 5: Starting GraphHopper service..."
echo "--------------------------------------------------------------------------------"
sudo systemctl start graphhopper
sleep 5

# 6. Test service
echo ""
echo "Step 6: Testing GraphHopper API..."
echo "--------------------------------------------------------------------------------"
if curl -s http://localhost:8080/health | grep -q "OK"; then
    echo "✓ GraphHopper is healthy and responding"
else
    echo "✗ GraphHopper not responding - check logs:"
    echo "  tail -f graphhopper_server.log"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ GraphHopper rebuild complete!"
echo "================================================================================"
echo ""
echo "Truck profiles updated:"
for profile in truck_diesel truck_ev truck_h2; do
    echo "  - $profile"
done
echo ""
echo "Server status: sudo systemctl status graphhopper"
echo "View logs:     tail -f graphhopper_server.log"
echo ""
