#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage:
  $0 --case=A

Currently supported cases:
  A
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

if [[ "$CASE" != "A" ]]; then
    echo "[ERROR] This runner currently supports only --case=A" >&2
    exit 1
fi

# Donde está el symlink dentro del track
LINK_PATH="${BASH_SOURCE[0]}"
LINK_DIR="$(cd "$(dirname "$LINK_PATH")" && pwd)"
TRACK_DIR="$(cd "$LINK_DIR/.." && pwd)"

# Donde está el script real en rtl/scripts/tb
REAL_PATH="$(readlink -f "$LINK_PATH")"
REAL_DIR="$(cd "$(dirname "$REAL_PATH")" && pwd)"

# rtl root deducido desde rtl/scripts/tb
RTL_ROOT="$(cd "$REAL_DIR/../.." && pwd)"

TCL_PATH="$REAL_DIR/tb_compute_Aij.tcl"

if [[ ! -f "$TCL_PATH" ]]; then
    echo "[ERROR] Tcl not found: $TCL_PATH" >&2
    exit 1
fi

# Vivado logs/journal del launcher
LAUNCH_LOG_DIR="$TRACK_DIR/simulation/logs"
mkdir -p "$LAUNCH_LOG_DIR"

TCL_WIN="$(wslpath -w "$TCL_PATH")"
TRACK_DIR_WIN="$(wslpath -w "$TRACK_DIR")"
RTL_ROOT_WIN="$(wslpath -w "$RTL_ROOT")"

LOG_WIN="$(wslpath -w "$LAUNCH_LOG_DIR/vivado_xsim_${CASE}.log")"
JOU_WIN="$(wslpath -w "$LAUNCH_LOG_DIR/vivado_xsim_${CASE}.jou")"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
\$ErrorActionPreference = 'Stop'
\$env:TRACK_DIR_WIN = '$TRACK_DIR_WIN'
\$env:RTL_ROOT_WIN  = '$RTL_ROOT_WIN'
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch -source '$TCL_WIN' -log '$LOG_WIN' -journal '$JOU_WIN'
exit \$LASTEXITCODE
"