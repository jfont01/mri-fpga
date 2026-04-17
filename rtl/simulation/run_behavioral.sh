#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCL_PATH="${1:-$THIS_DIR/tb_compute_Aij.tcl}"

LOG_DIR="$THIS_DIR/logs"
mkdir -p "$LOG_DIR"

TCL_WIN=$(wslpath -w "$TCL_PATH")
LOG_WIN=$(wslpath -w "$LOG_DIR/vivado_behavioral.log")
JOU_WIN=$(wslpath -w "$LOG_DIR/vivado_behavioral.jou")

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
  "& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch -source '$TCL_WIN' -log '$LOG_WIN' -journal '$JOU_WIN'"