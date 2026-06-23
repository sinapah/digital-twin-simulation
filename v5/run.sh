#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

VENV_DIR="../v4/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: v4 venv not found. Run v4/setup.sh first."
    exit 1
fi

PYTHON="$VENV_DIR/bin/python3"

echo "Starting Digital Twin Federated Learning Simulation..."
echo "Delay model: ${1:-kde}"
echo "Rounds: ${2:-100}"
echo ""

$PYTHON simulator.py --delay-model "${1:-kde}" --rounds "${2:-100}"

echo ""
echo "Simulation finished. See outputs/ for metrics."