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

echo "=== V6 Digital Twin Simulation ==="
echo ""

case "$MODE" in

  fit)
    echo "Fitting KDE and WGAN models from baseline arrival logs..."
    $PYTHON fit_delays.py --arrivals-dir outputs/baseline --out-dir ../v2
    ;;

  sender0|sender1|sender2)
    EDGE_ID="${MODE: -1}"
    UDP_PORT=$((7000 + EDGE_ID * 2))
    shift
    echo "Starting Sender ${EDGE_ID}..."
    echo "  UDP port: ${UDP_PORT}, control port: $((UDP_PORT + 1))"
    $PYTHON sender.py \
      --edge-id "${EDGE_ID}" \
      --udp-port "${UDP_PORT}" \
      "${@}"
    ;;

  baseline|kde|wgan)
    # Original v5 modes — DT disabled (backward compatible)
    if [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
      echo "Mode:         $MODE"
      echo "Rounds:       $ROUNDS"
      echo "Sender VMs:   $3 $4 $5"
      echo ""
      $PYTHON simulator.py \
        --mode "$MODE" \
        --rounds "$ROUNDS" \
        --dt-mode none \
        --sender-hosts "$3" "$4" "$5" \
        "${@:6}"
    else
      echo "Mode:   $MODE"
      echo "Rounds: $ROUNDS"
      echo "(local senders, DT disabled)"
      echo ""
      $PYTHON simulator.py \
        --mode "$MODE" \
        --rounds "$ROUNDS" \
        --dt-mode none \
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

  dt-baseline|dt-kde|dt-wgan)
    # New v6 DT-enabled modes
    DT_SYNTH="none"
    FALLBACK_MODE="kde"
    case "$MODE" in
      dt-baseline) DT_SYNTH="none";   FALLBACK_MODE="baseline" ;;
      dt-kde)      DT_SYNTH="kde";    FALLBACK_MODE="kde" ;;
      dt-wgan)     DT_SYNTH="wgan";   FALLBACK_MODE="wgan" ;;
    esac

    if [ -n "$3" ] && [ -n "$4" ] && [ -n "$5" ]; then
      echo "Mode:         $MODE"
      echo "DT synthesis: $DT_SYNTH"
      echo "Rounds:       $ROUNDS"
      echo "Sender VMs:   $3 $4 $5"
      echo ""
      $PYTHON simulator.py \
        --mode "$FALLBACK_MODE" \
        --rounds "$ROUNDS" \
        --dt-mode "$DT_SYNTH" \
        --sender-hosts "$3" "$4" "$5" \
        "${@:6}"
    else
      echo "Mode:         $MODE"
      echo "DT synthesis: $DT_SYNTH"
      echo "Rounds:       $ROUNDS"
      echo "(local senders)"
      echo ""
      $PYTHON simulator.py \
        --mode "$FALLBACK_MODE" \
        --rounds "$ROUNDS" \
        --dt-mode "$DT_SYNTH" \
        "${@:3}"
    fi

    echo ""
    echo "Simulation finished."
    echo "  Edge metrics:   outputs/$FALLBACK_MODE/"
    echo "  DT metrics:     outputs/dt/metrics/"
    echo "  DT history:     outputs/dt/history/"
    ;;

  *)
    echo "V6 Digital Twin Simulation"
    echo ""
    echo "Traditional modes (DT disabled):"
    echo "  ./run.sh baseline [rounds]                    # Collect real delays"
    echo "  ./run.sh fit                                  # Fit KDE/WGAN from baseline"
    echo "  ./run.sh kde [rounds]                         # KDE delays, no DT"
    echo "  ./run.sh wgan [rounds]                        # WGAN delays, no DT"
    echo ""
    echo "Digital Twin modes (DT enabled):"
    echo "  ./run.sh dt-baseline [rounds]                 # DT in observation mode"
    echo "  ./run.sh dt-kde [rounds]                      # DT + KDE telemetry synthesis"
    echo "  ./run.sh dt-wgan [rounds]                     # DT + WGAN telemetry synthesis"
    echo ""
    echo "Remote senders (add 3 IPs at end):"
    echo "  ./run.sh dt-kde 100 <host0> <host1> <host2>"
    echo ""
    echo "On each sender VM:"
    echo "  ./run.sh sender0 --mode kde --fps 500 --rounds 100"
    ;;

esac
