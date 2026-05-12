#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --case=Aij
  $0 --case=bi
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
        *)
            echo "[ERROR] Unknown argument: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CASE" ]]; then
    echo "[ERROR] Missing required argument --case" >&2
    usage
    exit 1
fi

if ! command -v vivado >/dev/null 2>&1; then
    echo "[ERROR] vivado not found in PATH" >&2
    exit 1
fi

TRACK_DIR="$(pwd -P)"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

FLIST_PATH="$TRACK_DIR/flist/tb_compute_${CASE}.flist"
TCL_PATH="$SCRIPT_DIR/tb_compute_${CASE}.tcl"
STIMULI_DIR="$TRACK_DIR/stimuli"
VECTORS_RTL_DIR="$TRACK_DIR/vectors/rtl"
LOG_DIR="$TRACK_DIR/simulation/logs"

case "$CASE" in
    Aij)
        OUT_FILE="$VECTORS_RTL_DIR/rtl_A.dat"
        ;;
    bi)
        OUT_FILE="$VECTORS_RTL_DIR/rtl_b.dat"
        ;;
    *)
        echo "[ERROR] Unsupported case: $CASE" >&2
        exit 1
        ;;
esac

[[ -f "$FLIST_PATH" ]]   || { echo "[ERROR] Missing flist: $FLIST_PATH" >&2; exit 1; }
[[ -f "$TCL_PATH" ]]     || { echo "[ERROR] Missing Tcl: $TCL_PATH" >&2; exit 1; }
[[ -d "$STIMULI_DIR" ]]  || { echo "[ERROR] Missing stimuli dir: $STIMULI_DIR" >&2; exit 1; }

mkdir -p "$VECTORS_RTL_DIR"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/vivado_xsim_${CASE}.log"
JOU_FILE="$LOG_DIR/vivado_xsim_${CASE}.jou"

echo "[run_xsim.sh] TRACK_DIR   = $TRACK_DIR"
echo "[run_xsim.sh] CASE        = $CASE"
echo "[run_xsim.sh] FLIST       = $FLIST_PATH"
echo "[run_xsim.sh] TCL         = $TCL_PATH"
echo "[run_xsim.sh] STIMULI_DIR = $STIMULI_DIR"
echo "[run_xsim.sh] OUT_FILE    = $OUT_FILE"

vivado \
  -mode batch \
  -source "$TCL_PATH" \
  -log "$LOG_FILE" \
  -journal "$JOU_FILE" \
  -tclargs "$TRACK_DIR" "$CASE" "$FLIST_PATH" "$STIMULI_DIR" "$OUT_FILE"