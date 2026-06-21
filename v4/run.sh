# =========================================================
# V4 Run Script
# =========================================================
# Example: Run the federated learning system
# =========================================================

#!/bin/bash

# Configuration
AGGREGATOR_VM_IP="10.181.160.135"
EDGE0_VM_IP="10.181.160.61"
EDGE1_VM_IP="10.181.160.167"
EDGE2_VM_IP="10.181.160.91"

# Ports
AGGREGATOR_PORT=5000
EDGE0_SENDER_PORT=6000
EDGE1_SENDER_PORT=6001
EDGE2_SENDER_PORT=6002

echo "=== V4 Federated Learning System ==="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

source venv/bin/activate

# Parse command line arguments
case "$1" in
    aggregator)
        echo "Starting Aggregator..."
        echo "  Port: $AGGREGATOR_PORT"
        python3 aggregator.py --port $AGGREGATOR_PORT
        ;;
    
    edge0)
        echo "Starting Edge 0..."
        echo "  Intersections: 0-7"
        echo "  Aggregator: $AGGREGATOR_VM_IP:$AGGREGATOR_PORT"
        echo "  Sender Port: $EDGE0_SENDER_PORT"
        python3 edge_agent.py \
            --edge-id 0 \
            --intersection-indices 0 1 2 3 4 5 6 7 \
            --aggregator-host $AGGREGATOR_VM_IP \
            --aggregator-port $AGGREGATOR_PORT \
            --sender-port $EDGE0_SENDER_PORT
        ;;
    
    edge1)
        echo "Starting Edge 1..."
        echo "  Intersections: 8-15"
        echo "  Aggregator: $AGGREGATOR_VM_IP:$AGGREGATOR_PORT"
        echo "  Sender Port: $EDGE1_SENDER_PORT"
        python3 edge_agent.py \
            --edge-id 1 \
            --intersection-indices 8 9 10 11 12 13 14 15 \
            --aggregator-host $AGGREGATOR_VM_IP \
            --aggregator-port $AGGREGATOR_PORT \
            --sender-port $EDGE1_SENDER_PORT
        ;;
    
    edge2)
        echo "Starting Edge 2..."
        echo "  Intersections: 16-23"
        echo "  Aggregator: $AGGREGATOR_VM_IP:$AGGREGATOR_PORT"
        echo "  Sender Port: $EDGE2_SENDER_PORT"
        python3 edge_agent.py \
            --edge-id 2 \
            --intersection-indices 16 17 18 19 20 21 22 23 \
            --aggregator-host $AGGREGATOR_VM_IP \
            --aggregator-port $AGGREGATOR_PORT \
            --sender-port $EDGE2_SENDER_PORT
        ;;
    
    sender0)
        echo "Starting Sender 0..."
        echo "  Target: $EDGE0_VM_IP:$EDGE0_SENDER_PORT"
        python3 sender.py \
            --edge-id 0 \
            --intersection-indices 0 1 2 3 4 5 6 7 \
            --target-host $EDGE0_VM_IP \
            --target-port $EDGE0_SENDER_PORT
        ;;
    
    sender1)
        echo "Starting Sender 1..."
        echo "  Target: $EDGE1_VM_IP:$EDGE1_SENDER_PORT"
        python3 sender.py \
            --edge-id 1 \
            --intersection-indices 8 9 10 11 12 13 14 15 \
            --target-host $EDGE1_VM_IP \
            --target-port $EDGE1_SENDER_PORT
        ;;
    
    sender2)
        echo "Starting Sender 2..."
        echo "  Target: $EDGE2_VM_IP:$EDGE2_SENDER_PORT"
        python3 sender.py \
            --edge-id 2 \
            --intersection-indices 16 17 18 19 20 21 22 23 \
            --target-host $EDGE2_VM_IP \
            --target-port $EDGE2_SENDER_PORT
        ;;
    
    help|--help|-h)
        echo "V4 Federated Learning System"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  aggregator    Start the federated learning aggregator"
        echo "  edge0         Start edge agent 0 (intersections 0-7)"
        echo "  edge1         Start edge agent 1 (intersections 8-15)"
        echo "  edge2         Start edge agent 2 (intersections 16-23)"
        echo "  sender0       Start sender for edge 0"
        echo "  sender1       Start sender for edge 1"
        echo "  sender2       Start sender for edge 2"
        echo "  help          Show this help message"
        echo ""
        echo "Example:"
        echo "  $0 aggregator"
        echo "  $0 edge0"
        echo "  $0 sender0"
        echo ""
        ;;
    
    *)
        echo "Error: Unknown command '$1'"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
