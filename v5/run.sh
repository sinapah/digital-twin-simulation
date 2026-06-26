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

echo "=== V5 Digital Twin Simulation ==="
echo ""

case "$MODE" in

  fit)
    echo "Fitting KDE and WGAN models from baseline arrival logs..."
    $PYTHON fit_delays.py --arrivals-dir outputs/baseline --out-dir ../v2
    ;;

  sender0|sender1|sender2)
    # Run a sender for a specific edge — used when sender is on a separate VM.
    # Usage: ./run.sh sender0 [--mode baseline|kde|wgan] [--fps 25] [--rounds 100]
    EDGE_ID="${MODE: -1}"
    UDP_PORT=$((7000 + EDGE_ID * 2))
    shift
    echo "Starting Sender ${EDGE_ID}..."
    echo "  UDP port: ${UDP_PORT}, control port: $((UDP_PORT + 1))"
    echo "  (Edge will connect to this machine's control port)"
    $PYTHON sender.py \
      --edge-id "${EDGE_ID}" \
      --udp-port "${UDP_PORT}" \
      "${@}"
    ;;

  baseline|kde|wgan)
    # Run the full simulation.
    # With remote senders: ./run.sh kde 100 <host0> <host1> <host2>
    # Without (local):     ./run.sh kde 100
    if [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
      echo "Mode:         $MODE"
      echo "Rounds:       $ROUNDS"
      echo "Sender VMs:   $3 $4 $5"
      echo ""
      $PYTHON simulator.py \
        --mode "$MODE" \
        --rounds "$ROUNDS" \
        --sender-hosts "$3" "$4" "$5" \
        "${@:6}"
    else
      echo "Mode:   $MODE"
      echo "Rounds: $ROUNDS"
      echo "(local senders)"
      echo ""
      $PYTHON simulator.py \
        --mode "$MODE" \
        --rounds "$ROUNDS" \
        "${@:3}"
    fi

    echo ""
    echo "Simulation finished. Metrics in outputs/$MODE/"
    if [ "$MODE" = "baseline" ]; then
      echo ""
      echo "Next step — fit KDE/WGAN models:"
      echo "  ./run.sh fit"
    fi
    ;;

  *)
    echo "Usage:"
    echo "  ./run.sh baseline [rounds]                          # Collect real delays"
    echo "  ./run.sh fit                                        # Fit KDE/WGAN from baseline"
    echo "  ./run.sh kde [rounds]                               # Simulate with KDE delays"
    echo "  ./run.sh wgan [rounds]                              # Simulate with WGAN delays"
    echo ""
    echo "  # With remote sender VMs:"
    echo "  ./run.sh baseline 100 <host0> <host1> <host2>"
    echo "  ./run.sh kde      100 <host0> <host1> <host2>"
    echo ""
    echo "  # On each sender VM:"
    echo "  ./run.sh sender0 --mode baseline --fps 25 --rounds 100"
    echo "  ./run.sh sender1 --mode baseline --fps 25 --rounds 100"
    echo "  ./run.sh sender2 --mode baseline --fps 25 --rounds 100"
    ;;

esac
