#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0
  $0 --case=all
  $0 --case=A
  $0 --case=b
  $0 --case=div_restoring
  $0 --case=D
  $0 --case=I
  $0 --case=L
  $0 --case=m_hat
  $0 --case=x
  $0 --case=z

Notes:
  - Default and --case=all compare only RTL-ready cases:
      A, b, div_restoring
  - D, I, L, m_hat, x and z are accepted explicitly, but they require
    corresponding RTL output files to exist.
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

TRACK_DIR="$(pwd -P)"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PY_VM_SCRIPT="$SCRIPT_DIR/run_vm.py"

if [[ ! -f "$PY_VM_SCRIPT" ]]; then
    echo "[ERROR] Python VM runner not found: $PY_VM_SCRIPT" >&2
    exit 1
fi

CMD=(python3 "$PY_VM_SCRIPT" --track-dir "$TRACK_DIR")

if [[ -n "$CASE" ]]; then
    CMD+=(--case "$CASE")
fi

echo "[run_vm.sh] TRACK_DIR = $TRACK_DIR"
echo "[run_vm.sh] SCRIPT    = $PY_VM_SCRIPT"
echo "[run_vm.sh] CASE      = ${CASE:-all}"

"${CMD[@]}"