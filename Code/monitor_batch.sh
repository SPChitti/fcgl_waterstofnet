#!/bin/bash
# Monitor batch route generation progress

BATCH_DIR="/home/ubuntu/fcgl_waterstofnet/Maps/batch_routes"
TOTAL_EXPECTED=306

while true; do
    clear
    echo "============================================================"
    echo "  BATCH ROUTE GENERATION - Progress Monitor"
    echo "============================================================"
    echo ""
    
    # Count JSON files (excluding summary/failed)
    CURRENT=$(ls "$BATCH_DIR"/*.json 2>/dev/null | grep -v "batch_summary" | grep -v "failed_pairs" | wc -l)
    PERCENT=$(echo "scale=1; ($CURRENT / $TOTAL_EXPECTED) * 100" | bc)
    
    echo "Progress: $CURRENT / $TOTAL_EXPECTED routes ($PERCENT%)"
    echo ""
    
    # Show progress bar
    FILLED=$(echo "scale=0; ($CURRENT * 50) / $TOTAL_EXPECTED" | bc)
    printf "["
    for i in $(seq 1 50); do
        if [ $i -le $FILLED ]; then
            printf "="
        else
            printf " "
        fi
    done
    printf "]\n\n"
    
    # Check if process is still running
    if ps aux | grep -q "[b]atch_generate_routes.py"; then
        echo "Status: ✓ RUNNING (PID: $(pgrep -f batch_generate_routes.py))"
    else
        echo "Status: ✗ STOPPED"
        echo ""
        echo "Batch generation complete or stopped!"
        break
    fi
    
    echo ""
    echo "Most recent routes:"
    ls -lt "$BATCH_DIR"/*.json 2>/dev/null | grep -v "batch_summary" | grep -v "failed_pairs" | head -5 | awk '{print "  " $9}' | xargs -I {} basename {}
    
    echo ""
    echo "Press Ctrl+C to exit monitor (generation continues in background)"
    
    sleep 10
done

echo ""
echo "Final count: $CURRENT routes generated"
echo "============================================================"
