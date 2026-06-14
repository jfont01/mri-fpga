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

run_case() {
    local CASE="$1"

    local SYNTH_PATH="$TRACK_DIR/synthesis/synth_$CASE"
    mkdir -p "$SYNTH_PATH"

    local LOG_DIR="$SYNTH_PATH/logs"
    mkdir -p "$LOG_DIR"

    echo "[run_synth.sh] Running from: $TRACK_DIR"
    echo "[run_synth.sh] CASE: $CASE"

    require_file "$RTL_SYNTH_TCL"

    vivado \
      -mode batch \
      -notrace \
      -source "$RTL_SYNTH_TCL" \
      -log "$LOG_DIR/vivado.log" \
      -journal "$LOG_DIR/vivado.jou" \
      -tclargs "$TRACK_DIR" "$CASE"
}

run_case_if_available() {
    local CASE="$1"

    case "$CASE" in
        div_restoring)
            if [[ ! -f "$TRACK_DIR/flist/synth_div_restoring.flist" ]]; then
                echo "[run_synth.sh] Skipping div_restoring: missing flist/synth_div_restoring.flist"
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

: "${RTL_SYNTH_TCL:?RTL_SYNTH_TCL is not defined. Did you run: source set_env.sh ?}"

VIVADO_SETTINGS="${VIVADO_SETTINGS:-/tools/Xilinx/Vivado/2024.2/settings64.sh}"

if [[ -f "$VIVADO_SETTINGS" ]]; then
    source "$VIVADO_SETTINGS"
fi

if ! command -v vivado >/dev/null 2>&1; then
    echo "[ERROR] vivado not found in PATH" >&2
    exit 1
fi

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