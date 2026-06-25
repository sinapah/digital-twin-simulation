#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

VENV_DIR="../v4/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: v4 venv not found. Run v4/setup.sh first."
    exit 1
fi

PYTHON="$VENV_DIR/bin/python3"
MODE="${1:-kde}"
ROUNDS="${2:-100}"

echo "Starting Digital Twin Federated Learning Simulation..."
echo "Mode:   $MODE"
echo "Rounds: $ROUNDS"
echo ""

if [ "$MODE" = "fit" ]; then
    echo "Fitting KDE and WGAN models from baseline arrival logs..."
    $PYTHON fit_delays.py --arrivals-dir outputs/baseline --out-dir ../v2
else
    $PYTHON simulator.py --mode "$MODE" --rounds "$ROUNDS" "${@:3}"
    echo ""
    echo "Simulation finished. Metrics in outputs/$MODE/"
    if [ "$MODE" = "baseline" ]; then
        echo ""
        echo "To fit KDE/WGAN from these delays, run:"
        echo "  ./run.sh fit"
    fi
fi
