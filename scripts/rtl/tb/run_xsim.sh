#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --case=Aij
  $0 --case=bi
  $0 --case=div_restoring
  $0 --case=all
EOF
}

require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Missing file: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"

    if [[ ! -d "$path" ]]; then
        echo "[ERROR] Missing directory: $path" >&2
        exit 1
    fi
}

find_div_restoring_mode() {
    local stimuli_dir="$1"

    shopt -s nullglob
    local files=( "$stimuli_dir"/div_restoring_*_in.dat )
    shopt -u nullglob

    if (( ${#files[@]} == 0 )); then
        echo "[ERROR] Missing div_restoring input file in: $stimuli_dir" >&2
        echo "[ERROR] Expected something like: div_restoring_trunc_in.dat" >&2
        exit 1
    fi

    if (( ${#files[@]} > 1 )); then
        echo "[ERROR] Multiple div_restoring input files found in: $stimuli_dir" >&2
        printf '  %s\n' "${files[@]}" >&2
        echo "[ERROR] Keep only one div_restoring_*_in.dat per track for now." >&2
        exit 1
    fi

    local base
    base="$(basename "${files[0]}")"

    local mode="${base#div_restoring_}"
    mode="${mode%_in.dat}"

    echo "$mode"
}

run_case() {
    local CASE="$1"

    local STIMULI_DIR="$TRACK_DIR/stimuli"
    local VECTORS_RTL_DIR="$TRACK_DIR/vectors/rtl"
    local LOG_DIR="$TRACK_DIR/simulation/logs"

    local FLIST_PATH
    local TCL_PATH
    local OUT_FILE

    case "$CASE" in
        Aij)
            FLIST_PATH="$TRACK_DIR/flist/tb_compute_Aij.flist"
            TCL_PATH="$SCRIPT_DIR/tb_compute_Aij.tcl"
            OUT_FILE="$VECTORS_RTL_DIR/rtl_A.dat"
            ;;

        bi)
            FLIST_PATH="$TRACK_DIR/flist/tb_compute_bi.flist"
            TCL_PATH="$SCRIPT_DIR/tb_compute_bi.tcl"
            OUT_FILE="$VECTORS_RTL_DIR/rtl_b.dat"
            ;;

        div_restoring)
            local DIV_MODE
            DIV_MODE="$(find_div_restoring_mode "$STIMULI_DIR")"

            FLIST_PATH="$TRACK_DIR/flist/tb_div_restoring.flist"
            TCL_PATH="$SCRIPT_DIR/tb_div_restoring.tcl"
            OUT_FILE="$VECTORS_RTL_DIR/rtl_div_restoring_${DIV_MODE}.dat"
            ;;

        *)
            echo "[ERROR] Unsupported case: $CASE" >&2
            exit 1
            ;;
    esac

    require_file "$FLIST_PATH"
    require_file "$TCL_PATH"
    require_dir "$STIMULI_DIR"

    mkdir -p "$VECTORS_RTL_DIR"
    mkdir -p "$LOG_DIR"

    local LOG_FILE="$LOG_DIR/vivado_xsim_${CASE}.log"
    local JOU_FILE="$LOG_DIR/vivado_xsim_${CASE}.jou"

    echo "[run_xsim.sh] TRACK_DIR   = $TRACK_DIR"
    echo "[run_xsim.sh] CASE        = $CASE"
    echo "[run_xsim.sh] FLIST       = $FLIST_PATH"
    echo "[run_xsim.sh] TCL         = $TCL_PATH"
    echo "[run_xsim.sh] STIMULI_DIR = $STIMULI_DIR"
    echo "[run_xsim.sh] OUT_FILE    = $OUT_FILE"

    vivado \
      -mode batch \
      -notrace \
      -source "$TCL_PATH" \
      -log "$LOG_FILE" \
      -journal "$JOU_FILE" \
      -tclargs "$TRACK_DIR" "$CASE" "$FLIST_PATH" "$STIMULI_DIR" "$OUT_FILE"
}

run_case_if_available() {
    local CASE="$1"

    case "$CASE" in
        div_restoring)
            if [[ ! -f "$TRACK_DIR/flist/tb_div_restoring.flist" ]]; then
                echo "[run_xsim.sh] Skipping div_restoring: missing flist/tb_div_restoring.flist"
                return 0
            fi
            ;;
    esac

    run_case "$CASE"
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

case "$CASE" in
    Aij|bi|div_restoring)
        run_case "$CASE"
        ;;

    all)
        run_case "Aij"
        run_case "bi"
        run_case_if_available "div_restoring"
        ;;

    *)
        echo "[ERROR] Unsupported case: $CASE" >&2
        usage
        exit 1
        ;;
esac