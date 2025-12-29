#!/bin/bash
# GraphHopper Management Script
# This is a convenience wrapper - systemd handles everything automatically

case "$1" in
    start)
        sudo systemctl start graphhopper
        echo "✓ GraphHopper started"
        ;;
    stop)
        sudo systemctl stop graphhopper
        echo "✓ GraphHopper stopped"
        ;;
    restart)
        sudo systemctl restart graphhopper
        echo "✓ GraphHopper restarted"
        ;;
    status)
        sudo systemctl status graphhopper --no-pager
        ;;
    logs)
        tail -f /home/ubuntu/fcgl_waterstofnet/Data/graphhopper_server.log
        ;;
    test)
        echo "Testing GraphHopper API..."
        curl -s http://localhost:8080/health && echo -e "\n✓ Server is healthy" || echo "✗ Server not responding"
        ;;
    *)
        echo "GraphHopper Management Script"
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start GraphHopper server"
        echo "  stop    - Stop GraphHopper server"
        echo "  restart - Restart GraphHopper server"
        echo "  status  - Show service status"
        echo "  logs    - Follow server logs (Ctrl+C to exit)"
        echo "  test    - Test if server is responding"
        echo ""
        echo "Note: GraphHopper auto-starts on EC2 reboot"
        exit 1
        ;;
esac
