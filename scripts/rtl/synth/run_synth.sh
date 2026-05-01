#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --case=Aij
  $0 --case=bi
  $0 Aij
  $0 bi
EOF
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
echo "[run_synth.sh] Running from: $TRACK_DIR"
echo "[run_synth.sh] CASE: $CASE"

SYNTH_PATH="$TRACK_DIR/synthesis/synth_$CASE"
mkdir -p $SYNTH_PATH

LOG_DIR="$SYNTH_PATH/logs"
mkdir -p "$LOG_DIR"

source /tools/Xilinx/Vivado/2024.2/settings64.sh
vivado -mode batch -source "$RTL_SYNTH_TCL" -log "$LOG_DIR/vivado.log" -journal "$LOG_DIR/vivado.jou" -tclargs "$TRACK_DIR" "$CASE"