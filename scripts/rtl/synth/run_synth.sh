#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --case=Aij
  $0 --case=bi
  $0 --case=all
EOF
}

run_case() {
    local CASE="$1"

    local SYNTH_PATH="$TRACK_DIR/synthesis/synth_$CASE"
    mkdir -p "$SYNTH_PATH"

    local LOG_DIR="$SYNTH_PATH/logs"
    mkdir -p "$LOG_DIR"

    echo "[run_synth.sh] Running from: $TRACK_DIR"
    echo "[run_synth.sh] CASE: $CASE"

    vivado \
      -mode batch \
      -source "$RTL_SYNTH_TCL" \
      -log "$LOG_DIR/vivado.log" \
      -journal "$LOG_DIR/vivado.jou" \
      -tclargs "$TRACK_DIR" "$CASE"
}

CASE=""

for arg in "$@"; do
    case "$arg" in
        --case=*)
            CASE="${arg#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $arg" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$CASE" ]]; then
                CASE="$arg"
            else
                echo "[ERROR] Unexpected extra argument: $arg" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$CASE" ]]; then
    echo "[ERROR] Missing case" >&2
    usage
    exit 1
fi

TRACK_DIR="$(pwd -P)"

source /tools/Xilinx/Vivado/2024.2/settings64.sh

case "$CASE" in
    Aij|bi)
        run_case "$CASE"
        ;;
    all)
        run_case "Aij"
        run_case "bi"
        ;;
    *)
        echo "[ERROR] Unsupported case: $CASE" >&2
        usage
        exit 1
        ;;
esac