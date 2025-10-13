#!/bin/bash

# SearXNG Docker Management Script

case "$1" in
    "start")
        echo "Starting SearXNG with Docker Compose..."
        docker compose up -d searxng redis
        echo "Waiting for SearXNG to be ready..."
        sleep 10
        echo "SearXNG should be available at http://localhost:8081"
        echo "You can test it with: python test_searx.py"
        ;;
    "stop")
        echo "Stopping SearXNG containers..."
        docker compose down
        ;;
    "logs")
        echo "Showing SearXNG logs..."
        docker compose logs -f searxng
        ;;
    "test")
        echo "Testing SearXNG connection..."
        curl -s "http://localhost:8081/search?q=test&format=json" | jq '.results | length' 2>/dev/null || echo "SearXNG not responding or jq not installed"
        ;;
    "restart")
        echo "Restarting SearXNG..."
        docker compose down
        docker compose up -d searxng redis
        echo "SearXNG restarted"
        ;;
    *)
        echo "Usage: $0 {start|stop|logs|test|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start SearXNG and Redis containers"
        echo "  stop    - Stop all containers"
        echo "  logs    - Show SearXNG logs"
        echo "  test    - Test if SearXNG is responding"
        echo "  restart - Restart SearXNG containers"
        exit 1
        ;;
esac