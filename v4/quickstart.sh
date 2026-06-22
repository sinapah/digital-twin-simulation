#!/bin/bash
# =========================================================
# V4 Quick Start Script
# =========================================================
# Starts all components (aggregator + 3 edges) on the local machine
# Use this for testing before moving to Multipass VMs
# =========================================================

set -e

echo "=== V4 Quick Start (Local Testing) ==="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Configuration
AGGREGATOR_PORT=5000
EDGE0_PORT=6000
EDGE1_PORT=6001
EDGE2_PORT=6002

# Use venv's python directly
PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"

# Use venv's python directly
PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"

echo "Starting Aggregator on port $AGGREGATOR_PORT..."
$PYTHON_CMD aggregator.py --port $AGGREGATOR_PORT &
AGGREGATOR_PID=$!

sleep 2

echo "Starting Edge 0..."
$PYTHON_CMD edge_agent.py \
    --edge-id 0 \
    --intersection-indices 0 1 2 3 4 5 6 7 \
    --aggregator-host localhost \
    --aggregator-port $AGGREGATOR_PORT \
    --sender-port $EDGE0_PORT &
EDGE0_PID=$!

sleep 1

echo "Starting Edge 1..."
$PYTHON_CMD edge_agent.py \
    --edge-id 1 \
    --intersection-indices 8 9 10 11 12 13 14 15 \
    --aggregator-host localhost \
    --aggregator-port $AGGREGATOR_PORT \
    --sender-port $EDGE1_PORT &
EDGE1_PID=$!

sleep 1

echo "Starting Edge 2..."
$PYTHON_CMD edge_agent.py \
    --edge-id 2 \
    --intersection-indices 16 17 18 19 20 21 22 23 \
    --aggregator-host localhost \
    --aggregator-port $AGGREGATOR_PORT \
    --sender-port $EDGE2_PORT &
EDGE2_PID=$!

echo ""
echo "=== All processes started ==="
echo "  Aggregator PID: $AGGREGATOR_PID"
echo "  Edge 0 PID: $EDGE0_PID"
echo "  Edge 1 PID: $EDGE1_PID"
echo "  Edge 2 PID: $EDGE2_PID"
echo ""
echo "Press Ctrl+C to stop all processes"
echo ""
echo "=== Training Log (timestamps will be added by processes) ==="
echo ""

# Wait for user to stop
trap "kill $AGGREGATOR_PID $EDGE0_PID $EDGE1_PID $EDGE2_PID 2>/dev/null; echo ''; echo 'Stopped all processes'; exit 0" INT

# Keep running
wait
